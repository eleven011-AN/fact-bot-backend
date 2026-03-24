import os
import sys
import json
import re
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import ClaimStatus, Source  # type: ignore


def _generate_with_retry(model, prompt, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt)
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "ResourceExhausted" in err_str or "quota" in err_str.lower():
                if attempt < max_retries - 1:
                    wait = 15 * (attempt + 1)
                    print(f"Rate limit, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
            raise


def verify_claim(claim_id: int, claim_text: str, evidence: list, language: str = 'English') -> ClaimStatus:
    import google.generativeai as genai  # type: ignore

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    evidence_text = "\n\n".join([
        f"Source: {e.get('url', e.get('link', 'Unknown'))}\nSnippet: {e.get('snippet', e.get('content', ''))}"
        for e in evidence
    ])

    if not evidence_text.strip():
        explanation = "The search agent could not find any live web evidence related to this claim."
        return ClaimStatus(
            id=claim_id, text=claim_text,
            verdict="Unverifiable", confidence=0.0,
            explanation=explanation, sources=[]
        )

    lang_rule = ""
    if language and language != 'English':
        lang_rule = (
            f"\n\n⚠️ CRITICAL: Write the 'explanation' field entirely in {language}. "
            f"The 'verdict' must stay in English (True / False / Partially True / Unverifiable)."
        )

    prompt = (
        f"You are an expert fact-checker.{lang_rule}\n\n"
        "Compare the CLAIM against the EVIDENCE and classify it as: True, False, Partially True, or Unverifiable.\n"
        "Output ONLY a JSON object with exactly these keys: 'verdict', 'confidence', 'explanation'.\n"
        "Rules:\n"
        "- 'verdict': always in English — one of: True, False, Partially True, Unverifiable\n"
        "- 'confidence': a float between 0.0 and 1.0\n"
        f"- 'explanation': {'MUST be in ' + language if language and language != 'English' else 'clear English explanation'}\n"
        '- Example: {"verdict": "True", "confidence": 0.95, "explanation": "..."}\n\n'
        f"CLAIM: {claim_text}\n\nEVIDENCE:\n{evidence_text}"
    )

    try:
        response = _generate_with_retry(model, prompt)
        content = response.text
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            verdict = data.get("verdict", "Unverifiable")
            confidence = data.get("confidence", 0.0)
            explanation = data.get("explanation", "Could not parse explanation.")
        else:
            verdict, confidence, explanation = "Unverifiable", 0.0, "Failed to parse JSON response."
    except Exception as e:
        print(f"Verification error: {e}")
        verdict, confidence, explanation = "Unverifiable", 0.0, f"Error: {e}"

    sources = [
        Source(url=e.get("url", e.get("link", "")), title=e.get("title", "Source Document"))
        for e in evidence if (e.get("url") or e.get("link"))
    ]

    return ClaimStatus(
        id=claim_id, text=claim_text,
        verdict=verdict, confidence=confidence,
        explanation=explanation, sources=sources
    )
