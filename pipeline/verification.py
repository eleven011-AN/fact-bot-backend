import os
import sys
import json
import re
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import ClaimStatus, Source  # type: ignore


def verify_claim(claim_id: int, claim_text: str, evidence: list, language: str = 'English') -> ClaimStatus:
    from openai import OpenAI  # type: ignore

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)

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

    system = (
        f"You are an expert fact-checker.{lang_rule}\n\n"
        "Compare the CLAIM against the EVIDENCE and classify it.\n"
        "Output ONLY a JSON object with exactly these keys:\n"
        "- 'verdict': one of: True, False, Partially True, Unverifiable (always in English)\n"
        "- 'confidence': float 0.0-1.0\n"
        f"- 'explanation': {'MUST be in ' + language if language and language != 'English' else 'clear English explanation'}\n"
        'Example: {"verdict": "True", "confidence": 0.95, "explanation": "..."}'
    )
    user = f"CLAIM: {claim_text}\n\nEVIDENCE:\n{evidence_text}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0,
            max_tokens=1000,
        )
        content = response.choices[0].message.content or ""
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
