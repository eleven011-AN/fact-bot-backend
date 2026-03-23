import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import ClaimStatus, Source  # type: ignore

def verify_claim(claim_id: int, claim_text: str, evidence: list, language: str = 'English') -> ClaimStatus:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    from langchain_core.prompts import ChatPromptTemplate  # type: ignore
    import json
    import re
    
    api_key = os.getenv("GOOGLE_API_KEY")
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)
    
    evidence_text = "\n\n".join([
        f"Source: {e.get('url', e.get('link', 'Unknown'))}\nSnippet: {e.get('snippet', e.get('content', ''))}"
        for e in evidence
    ])
    
    if not evidence_text.strip():
        if language and language != 'English':
            # Ask the LLM to write the fallback explanation in the target language
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
                _llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=api_key, temperature=0)
                fallback_prompt = f"Translate this sentence to {language}: 'The search agent could not find any live web evidence related to this claim.'"
                _resp = _llm.invoke(fallback_prompt)
                explanation = _resp.content.strip()
            except Exception:
                explanation = f"({language}) No evidence found for this claim."
        else:
            explanation = "The search agent could not find any live web evidence related to this claim."
        return ClaimStatus(
            id=claim_id,
            text=claim_text,
            verdict="Unverifiable",
            confidence=0.0,
            explanation=explanation,
            sources=[]
        )
    
    # Build a very explicit language constraint
    if language and language != 'English':
        lang_rule = (
            f"\n\n⚠️ CRITICAL LANGUAGE RULE: You MUST write the 'explanation' field entirely in {language}. "
            f"Do NOT use English in the explanation. The explanation must be in {language} only. "
            f"The 'verdict' field must remain in English (True / False / Partially True / Unverifiable)."
        )
    else:
        lang_rule = ""

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         f"You are an expert fact-checker.{lang_rule}\n\n"
         "Compare the CLAIM against the EVIDENCE and classify it as: True, False, Partially True, or Unverifiable.\n"
         "Output ONLY a JSON object with exactly these keys: 'verdict', 'confidence', 'explanation'.\n"
         "Rules:\n"
         "- 'verdict': always in English — one of: True, False, Partially True, Unverifiable\n"
         "- 'confidence': a float between 0.0 and 1.0\n"
         f"- 'explanation': {'MUST be written in ' + language + '. Do NOT use English.' if language and language != 'English' else 'a clear English explanation'}\n"
         "Example: {{\"verdict\": \"True\", \"confidence\": 0.95, \"explanation\": \"<explanation in the required language here>\"}}"),
        ("user", "CLAIM: {claim}\n\nEVIDENCE:\n{evidence}")
    ])
    
    chain = prompt | llm
    try:
        response = chain.invoke({"claim": claim_text, "evidence": evidence_text})
        content = response.content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
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
        id=claim_id,
        text=claim_text,
        verdict=verdict,
        confidence=confidence,
        explanation=explanation,
        sources=sources
    )
