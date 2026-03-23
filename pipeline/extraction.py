import os
from typing import List


class RateLimitError(Exception):
    """Raised when the Gemini API quota / rate limit is exceeded."""
    pass


def extract_claims(text: str, language: str = 'English') -> List[str]:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    from langchain_core.prompts import ChatPromptTemplate  # type: ignore
    import json
    import re

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set. Please set it in .env")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key, temperature=0)

    lang_instruction = (
        f"IMPORTANT: Write your output JSON values in {language}."
        if language and language != 'English' else ""
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an expert fact-checker. The input text may be in any language — understand it fully regardless. "
         "Decompose the following text into distinct, independent, verifiable factual claims.\n"
         f"{lang_instruction}\n"
         "Output your response ONLY as a JSON object with a 'claims' key containing a list of strings.\n"
         "Example: {{\"claims\": [\"Claim 1\", \"Claim 2\"]}}"),
        ("user", "{text}")
    ])

    chain = prompt | llm
    try:
        response = chain.invoke({"text": text})
        content = response.content

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return data.get("claims", [])
            except json.JSONDecodeError as je:
                print(f"JSON Decode Error: {je}")
                return []
        else:
            print(f"No JSON found in LLM response: {content}")
            return []
    except RateLimitError:
        raise  # bubble up without wrapping
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "ResourceExhausted" in err_str or "quota" in err_str.lower():
            raise RateLimitError(
                "Gemini API quota exceeded. You've hit the free-tier rate limit. "
                "Please wait a minute and try again, or check your Google AI Studio quota."
            )
        print(f"Extraction error: {e}")
        return []

def extract_claims_from_image(base64_data: str, language: str = 'English') -> List[str]:
    from langchain_google_genai import ChatGoogleGenerativeAI  # type: ignore
    from langchain_core.messages import HumanMessage, SystemMessage  # type: ignore
    import json
    import re

    # Parse and strip the prepended 'data:image/jpeg;base64,' prefix 
    if "," in base64_data:
        mime_type = base64_data.split(';')[0].split(':')[1]
        base64_str = base64_data.split(',')[1]
    else:
        mime_type = "image/jpeg"
        base64_str = base64_data

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY is not set. Please set it in .env")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", google_api_key=api_key, temperature=0)

    lang_instruction = (
        f"IMPORTANT: Write your output JSON values in {language}."
        if language and language != 'English' else ""
    )

    sys_msg = SystemMessage(content=(
        "You are an expert fact-checker. Extract distinct, independent, verifiable factual claims from the text visible in this image. "
        f"{lang_instruction}\n"
        "Output ONLY a JSON object with a 'claims' key containing a list of strings.\n"
        "Example: {{\"claims\": [\"Claim 1\"]}}"
    ))

    # Send the image base64 directly to the multimodal model
    user_msg = HumanMessage(content=[
        {"type": "text", "text": "Extract claims from this image."},
        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_str}"}}
    ])

    try:
        response = llm.invoke([sys_msg, user_msg])
        content = response.content

        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(0))
                return data.get("claims", [])
            except json.JSONDecodeError as je:
                print(f"JSON Decode Error (Image): {je}")
                return []
        return []
    except RateLimitError:
        raise
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "ResourceExhausted" in err_str or "quota" in err_str.lower():
            raise RateLimitError(
                "Gemini API quota exceeded. You've hit the free-tier rate limit. "
                "Please wait a minute and try again."
            )
        print(f"Image extraction error: {e}")
        return []
