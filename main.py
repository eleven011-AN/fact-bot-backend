from fastapi import FastAPI, HTTPException  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from dotenv import load_dotenv  # type: ignore
import os
import io
import base64
import requests  # type: ignore
from bs4 import BeautifulSoup  # type: ignore
from models import VerifyRequest, VerifyResponse, ClaimStatus  # type: ignore
from pipeline.extraction import extract_claims, extract_claims_from_image, RateLimitError  # type: ignore
from pipeline.search import retrieve_evidence  # type: ignore
from pipeline.verification import verify_claim  # type: ignore
import asyncio
import time
from typing import List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pydantic import BaseModel  # type: ignore


load_dotenv()

app = FastAPI(title="Fact & Claim Verification API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def fetch_url_text(url: str) -> str:
    """Fetch article text from a URL, following redirects and stripping HTML."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        response = requests.get(url, timeout=15, headers=headers, allow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        # Remove script/style noise
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Try article body first, then fall back to all paragraphs
        article = soup.find('article')
        if article:
            text = article.get_text(separator=" ", strip=True)
        else:
            tags = soup.find_all(['p', 'h1', 'h2', 'h3', 'li'])
            text = " ".join(t.get_text(strip=True) for t in tags)

        text = text.strip()
        if not text or len(text) < 50:
            raise ValueError(
                "The page at that URL appears to be empty, blocked, or a redirect. "
                "Please paste the actual article URL (not a Bing/Google search result link)."
            )
        return text[:10000]  # Limit to 10k chars
    except ValueError:
        raise
    except Exception as e:
        raise ValueError(f"Could not read the URL — {e}")


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Fact Checking API is running!"}

@app.get("/api/debug-env")
def debug_env():
    """Check if API keys are loaded correctly on Render."""
    google_key = os.getenv("GOOGLE_API_KEY", "")
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    return {
        "google_key_set": bool(google_key),
        "google_key_preview": google_key[:6] + "..." if google_key else "NOT SET",
        "tavily_key_set": bool(tavily_key),
        "tavily_key_preview": tavily_key[:6] + "..." if tavily_key else "NOT SET",
    }

@app.get("/api/test-gemini")
def test_gemini():
    """Directly test the Gemini API key."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {"success": False, "error": "GOOGLE_API_KEY not set"}
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=api_key, temperature=0)
        resp = llm.invoke("Say OK in one word")
        return {"success": True, "response": resp.content}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ─── Live News Endpoint (GNews API) ───────────────────────────────────────────
NEWS_CACHE = {}
CACHE_TTL = 1800 # 30 mins

MOCK_ALL_ITEMS = [
  {"category": "Top Stories", "title": "Rep. Dan Goldman spoke in Congress about allegation that Trump sexually assaulted teen girl", "author": "Jack Izzo", "date": "March 20, 2026", "image": "https://images.unsplash.com/photo-1555848962-6e79363ec58f?auto=format&fit=crop&q=80&w=1000", "textToVerify": "Rep. Dan Goldman spoke in Congress about an allegation that Trump sexually assaulted a teen girl."},
  {"category": "Politics", "title": "21 rumors about Gavin Newsom we've looked into", "author": "Jordan Liles", "date": "March 18, 2026", "image": "https://images.unsplash.com/photo-1560250097-0b93528c311a?auto=format&fit=crop&q=80&w=300", "textToVerify": "Gavin Newsom is considering stepping down as governor to run a federal agency."},
  {"category": "World News", "title": "Viral video of military jets over capital is actually from a 2018 airshow", "author": "Defense Desk", "date": "March 21, 2026", "image": "https://images.unsplash.com/photo-1517976487492-5750f3195933?auto=format&fit=crop&q=80&w=1000", "textToVerify": "Viral video shows military jets flying low over the capital city during recent conflict."},
  {"category": "Sports", "title": "Rumor that LeBron James is retiring after this season is entirely fabricated", "author": "Sports Insider", "date": "March 22, 2026", "image": "https://images.unsplash.com/photo-1519861531473-9200262188bf?auto=format&fit=crop&q=80&w=1000", "textToVerify": "LeBron James announced he is officially retiring from basketball after this current season."},
  {"category": "Odd", "title": "Rumor that Florida man was kidnapped by dolphins is all wet", "author": "Staff", "date": "March 10, 2026", "image": "https://images.unsplash.com/photo-1599839619722-39751411ea63?auto=format&fit=crop&q=80&w=300", "textToVerify": "A Florida man was kidnapped by a pod of wild dolphins and held hostage."}
]

FRONTEND_CAT_MAP = {
    "Top Stories": "general",
    "Politics": "nation",
    "World News": "world",
    "Sports": "sports",
    "Odd": "health"
}

def free_translate(text: str, target_lang: str) -> str:
    if not text or target_lang == 'en': return text
    try:
        url = "https://translate.googleapis.com/translate_a/single"
        params = {"client": "gtx", "sl": "auto", "tl": target_lang, "dt": "t", "q": text}
        res = requests.get(url, params=params, timeout=5)
        data = res.json()
        return "".join([x[0] for x in data[0]]) if data and data[0] else text
    except:
        return text

def free_translate_concurrent(texts: List[str], target_lang: str) -> List[str]:
    if target_lang == 'en': return texts
    with ThreadPoolExecutor(max_workers=10) as executor:
        return list(executor.map(lambda t: free_translate(t, target_lang), texts))

class TranslateRequest(BaseModel):
    texts: List[str]
    target_lang: str

@app.post("/api/translate")
def translate_texts(req: TranslateRequest):
    translated = free_translate_concurrent(req.texts, req.target_lang)
    return {"translated": translated}

@app.get("/api/news")
def get_live_news(category: str = "Top Stories", lang: str = 'en'):
    cat = FRONTEND_CAT_MAP.get(category, "general")
    now = time.time()
    
    def get_en_articles():
        if category in NEWS_CACHE and (now - NEWS_CACHE[category]['timestamp']) < CACHE_TTL:
            return NEWS_CACHE[category]['data'], True, False
            
        api_key = os.getenv("GNEWS_API_KEY")
        if not api_key:
            filtered = MOCK_ALL_ITEMS if category == "Top Stories" else [i for i in MOCK_ALL_ITEMS if i.get('category') == category]
            if not filtered and MOCK_ALL_ITEMS: filtered = [MOCK_ALL_ITEMS[0]]
            return filtered, False, True
            
        try:
            url = f"https://gnews.io/api/v4/top-headlines?category={cat}&lang=en&country=us&max=10&apikey={api_key}"
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            articles = []
            for a in data.get("articles", []):
                pub_date = a.get('publishedAt', '')
                try:
                    dt = datetime.strptime(pub_date, "%Y-%m-%dT%H:%M:%SZ")
                    date_str = dt.strftime("%B %d, %Y")
                except:
                    date_str = "Recently"
                    
                articles.append({
                    "category": category,
                    "title": a.get("title", ""),
                    "author": a.get("source", {}).get("name", "News Desk"),
                    "date": date_str,
                    "image": a.get("image", "https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&q=80&w=1000"),
                    "textToVerify": a.get("title", "")
                })
                
            if not articles:
                raise ValueError("No articles found")
                
            NEWS_CACHE[category] = {'timestamp': now, 'data': articles}
            return articles, False, False
        except Exception as e:
            print(f"GNews Error: {e}")
            filtered = MOCK_ALL_ITEMS if category == "Top Stories" else [i for i in MOCK_ALL_ITEMS if i.get('category') == category]
            if not filtered and MOCK_ALL_ITEMS: filtered = [MOCK_ALL_ITEMS[0]]
            return filtered, False, True

    en_articles, cached, is_mock = get_en_articles()
    
    if lang == 'en':
        return {"articles": en_articles, "category": category, "cached": cached, "mock": is_mock}
        
    texts_to_translate = []
    for a in en_articles:
        texts_to_translate.extend([a['title'], a['textToVerify']])
        
    translated_texts = free_translate_concurrent(texts_to_translate, lang)
    
    translated_articles = []
    idx = 0
    for a in en_articles:
        new_a = dict(a)
        new_a['title'] = translated_texts[idx]
        new_a['textToVerify'] = translated_texts[idx+1]
        translated_articles.append(new_a)
        idx += 2
        
    return {"articles": translated_articles, "category": category, "cached": cached, "mock": is_mock}

# ─── TTS Endpoint ─────────────────────────────────────────────────────────────
class TTSRequest(BaseModel):
    text: str
    lang: str = 'en'  # BCP-47 language code prefix, e.g. 'hi', 'kn', 'ta'

@app.post("/api/tts")
async def text_to_speech(req: TTSRequest):
    """Generate speech audio using gTTS (Google TTS) for any supported language."""
    try:
        from gtts import gTTS  # type: ignore
        # gTTS uses language codes like 'hi', 'kn', 'ta', 'te', 'ml', 'en', 'fr', etc.
        lang_code = req.lang.split('-')[0]  # strip region suffix: 'hi-IN' -> 'hi'
        tts = gTTS(text=req.text[:1000], lang=lang_code, slow=False)
        mp3_fp = io.BytesIO()
        tts.write_to_fp(mp3_fp)
        mp3_fp.seek(0)
        audio_b64 = base64.b64encode(mp3_fp.read()).decode('utf-8')
        return {"audio": audio_b64, "format": "mp3"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {e}")

@app.post("/api/verify", response_model=VerifyResponse)
async def verify_content(request: VerifyRequest):
    if not os.getenv("GOOGLE_API_KEY"):
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not configured on the server.")
    try:
        if request.type == 'image':
            if not request.value.strip():
                raise HTTPException(status_code=400, detail="Image data is empty.")
            raw_claims = extract_claims_from_image(request.value, language=request.language or 'English')
        else:
            text_to_analyze = request.value
            if request.type == 'url':
                try:
                    text_to_analyze = fetch_url_text(request.value)
                except ValueError as e:
                    raise HTTPException(status_code=400, detail=str(e))
                    
            if not text_to_analyze.strip():
                raise HTTPException(status_code=400, detail="Text content is empty.")
                
            raw_claims = extract_claims(text_to_analyze, language=request.language or 'English')
            
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Claim extraction failed: {e}")
        
    if not raw_claims:
        return VerifyResponse(claims=[])

    def process_claim(args):
        idx, claim = args
        evidence = retrieve_evidence(claim)
        return verify_claim(idx, claim, evidence, language=request.language or 'English')
        
    def run_pipeline() -> list:  # type: ignore[return]
        return list(pool.map(process_claim, enumerate(raw_claims, start=1)))

    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=5) as pool:
        verified_claims = await loop.run_in_executor(pool, run_pipeline)  # type: ignore[arg-type]
        
    return VerifyResponse(claims=verified_claims)
