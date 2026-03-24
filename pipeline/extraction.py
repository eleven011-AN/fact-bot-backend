import os
import json
import re
import time
from typing import List


class RateLimitError(Exception):
    """Raised when the Gemini API quota / rate limit is exceeded."""
    pass


def _get_model():
    """Return a configured Gemini 2.0 Flash GenerativeModel."""
    import google.generativeai as genai  # type: ignore
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set. Please set it in .env")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.0-flash")


def _generate_with_retry(model, prompt, max_retries: int = 3):
    """Call model.generate_content with exponential backoff on 429 errors."""
    for attempt in range(max_retries):
        try:
            return model.generate_content(prompt)
        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "ResourceExhausted" in err_str or "quota" in err_str.lower()
            if is_rate_limit:
                if attempt < max_retries - 1:
                    wait = 15 * (attempt + 1)  # 15s, 30s, 45s
                    print(f"Rate limit hit, retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise RateLimitError(
                    "Gemini API quota exceeded. Please wait a minute and try again."
                )
            raise


def _parse_claims_json(content: str) -> List[str]:
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            return data.get("claims", [])
        except json.JSONDecodeError:
            pass
    return []


def extract_claims(text: str, language: str = 'English') -> List[str]:
    lang_instruction = (
        f"IMPORTANT: Write your output JSON values in {language}."
        if language and language != 'English' else ""
    )
    prompt = (
        "You are an expert fact-checker. Decompose the following text into distinct, "
        "independent, verifiable factual claims.\n"
        f"{lang_instruction}\n"
        "Output ONLY a JSON object with a 'claims' key containing a list of strings.\n"
        'Example: {"claims": ["Claim 1", "Claim 2"]}\n\n'
        f"TEXT:\n{text}"
    )
    try:
        model = _get_model()
        response = _generate_with_retry(model, prompt)
        return _parse_claims_json(response.text)
    except RateLimitError:
        raise
    except Exception as e:
        print(f"Extraction error: {e}")
        return []


def extract_claims_from_image(base64_data: str, language: str = 'English') -> List[str]:
    import google.generativeai as genai  # type: ignore
    import base64 as b64mod

    if "," in base64_data:
        mime_type = base64_data.split(';')[0].split(':')[1]
        base64_str = base64_data.split(',')[1]
    else:
        mime_type = "image/jpeg"
        base64_str = base64_data

    lang_instruction = (
        f"IMPORTANT: Write your output JSON values in {language}."
        if language and language != 'English' else ""
    )
    prompt = (
        "You are an expert fact-checker. Extract distinct, verifiable factual claims "
        "from the text visible in this image. "
        f"{lang_instruction}\n"
        "Output ONLY a JSON object with a 'claims' key containing a list of strings.\n"
        '{"claims": ["Claim 1"]}'
    )

    try:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is not set.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        image_part = {"mime_type": mime_type, "data": b64mod.b64decode(base64_str)}
        response = _generate_with_retry(model, [prompt, image_part])
        return _parse_claims_json(response.text)
    except RateLimitError:
        raise
    except Exception as e:
        print(f"Image extraction error: {e}")
        return []
