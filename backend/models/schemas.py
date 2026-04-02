from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class ModelInfo(BaseModel):
    id: str
    name: str
    description: str
    available: bool


class VQAQuery(BaseModel):
    question: str
    image_path: str
    model_id: Optional[str] = "vilt_curriculum"


class VQAResponse(BaseModel):
    answer: str
    confidence: float
    candidate_answers: List[Dict[str, Any]]

