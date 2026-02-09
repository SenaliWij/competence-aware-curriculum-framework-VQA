from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class ModelConfig(BaseModel):
    model_name: str
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 15
    use_curriculum: bool = True

class TrainingMetrics(BaseModel):
    epoch: int
    global_step: int
    loss: float
    accuracy: float
    current_tier: str
    tier_progress: float  # 0.0 to 1.0

class TrainingStatus(BaseModel):
    is_training: bool
    status_message: str
    metrics: TrainingMetrics

class VQAQuery(BaseModel):
    question: str
    image_id: Optional[str] = None

class VQAResponse(BaseModel):
    answer: str
    confidence: float
    candidate_answers: List[Dict[str, Any]]
    reasoning_trace: List[str]
