from fastapi import APIRouter, UploadFile, File
from models.schemas import ModelConfig, VQAQuery, VQAResponse, TrainingStatus
from services.training_service import training_service
from services.inference_service import inference_service

router = APIRouter()

@router.post("/training/start")
async def start_training(config: ModelConfig):
    return training_service.start_training(config)

@router.post("/training/stop")
async def stop_training():
    return training_service.stop_training()

@router.get("/training/status", response_model=TrainingStatus)
async def get_training_status():
    return training_service.get_status()

@router.post("/inference/predict", response_model=VQAResponse)
async def predict(query: VQAQuery):
    return inference_service.predict(query)

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # In a real app, save the file. For prototype, we just acknowledge receipt.
    return {"filename": file.filename, "message": "File uploaded successfully"}
