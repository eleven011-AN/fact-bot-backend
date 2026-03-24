import os
import json
import re
from typing import List


class RateLimitError(Exception):
    """Raised when the OpenAI API quota / rate limit is exceeded."""
    pass


def _get_client():
    from openai import OpenAI  # type: ignore
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Please set it in .env or Render environment.")
    return OpenAI(api_key=api_key)


def _chat(client, system: str, user: str) -> str:
    """Call OpenAI chat completion and return the text response."""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=2000,
        )
        return response.choices[0].message.content or ""
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "quota" in err_str.lower() or "rate_limit" in err_str.lower():
            raise RateLimitError(
                "OpenAI API quota exceeded. Please wait a moment and try again."
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
    system = (
        "You are an expert fact-checker. The input text may be in any language. "
        "Decompose the text into distinct, independent, verifiable factual claims.\n"
        f"{lang_instruction}\n"
        "Output ONLY a JSON object: {\"claims\": [\"Claim 1\", \"Claim 2\"]}"
    )
    try:
        client = _get_client()
        content = _chat(client, system, text)
        return _parse_claims_json(content)
    except RateLimitError:
        raise
    except Exception as e:
        print(f"Extraction error: {e}")
        return []


def extract_claims_from_image(base64_data: str, language: str = 'English') -> List[str]:
    from openai import OpenAI  # type: ignore

    if "," in base64_data:
        base64_str = base64_data  # keep full data URL for OpenAI
    else:
        base64_str = f"data:image/jpeg;base64,{base64_data}"

    lang_instruction = (
        f"IMPORTANT: Write your output JSON values in {language}."
        if language and language != 'English' else ""
    )
    system_prompt = (
        "You are an expert fact-checker. Extract distinct, verifiable factual claims "
        "from the text visible in this image. "
        f"{lang_instruction}\n"
        "Output ONLY a JSON object: {\"claims\": [\"Claim 1\"]}"
    )

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": system_prompt},
                    {"type": "image_url", "image_url": {"url": base64_str}},
                ],
            }],
            temperature=0,
            max_tokens=2000,
        )
        content = response.choices[0].message.content or ""
        return _parse_claims_json(content)
    except RateLimitError:
        raise
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "quota" in err_str.lower():
            raise RateLimitError("OpenAI API quota exceeded.")
        print(f"Image extraction error: {e}")
        return []
