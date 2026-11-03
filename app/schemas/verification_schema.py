from pydantic import BaseModel
from typing import Optional, Dict, Any

class VerificationResponse(BaseModel):
    status: str
    verdict: Optional[str] = None
    confidence_score: Optional[float] = None
    barriers_results: Dict[str, Any] = {}
    message: str