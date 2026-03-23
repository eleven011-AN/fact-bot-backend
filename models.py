from pydantic import BaseModel  # type: ignore
from typing import List, Optional

class VerifyRequest(BaseModel):
    type: str  # 'text' or 'url'
    value: str
    language: Optional[str] = 'English'  # output language for LLM responses

class Source(BaseModel):
    url: str
    title: str

class ClaimStatus(BaseModel):
    id: int
    text: str
    verdict: str  # 'True', 'False', 'Partially True', 'Unverifiable'
    confidence: float
    explanation: str
    sources: List[Source]

class VerifyResponse(BaseModel):
    claims: List[ClaimStatus]
