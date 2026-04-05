from pydantic import BaseModel
from typing import List, Dict, Any
#Request schema for VQA inference
class VQAQuery(BaseModel):
    question: str
    image_path: str

#Response schema for VQA inference
class VQAResponse(BaseModel):
    answer: str
    confidence: float
    candidate_answers: List[Dict[str, Any]]

#Request schema for VQA comparison
class VQACompareQuery(BaseModel):
    question: str
    image_path: str

#Response schema for VQA comparison
class VQAModelResult(BaseModel):
    model_id: str
    model_name: str
    answer: str
    confidence: float
    candidate_answers: List[Dict[str, Any]]

#Response schema for VQA comparison
class VQACompareResponse(BaseModel):
    proposed: VQAModelResult
    baseline: VQAModelResult
