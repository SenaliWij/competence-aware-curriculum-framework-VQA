import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import List
from models.schemas import ModelInfo, VQAQuery, VQAResponse
from services.model_service import model_service
from services.inference_service import inference_service

router = APIRouter()


# ── Models ────────────────────────────────────────────────────────────────────

@router.get("/models", response_model=List[ModelInfo])
async def list_models():
    return model_service.list_models()


@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    model = model_service.get_model(model_id)
    if not model:
        raise HTTPException(status_code=404, detail="Model not found")
    return model


@router.get("/models/{model_id}/download")
async def download_model(model_id: str):
    checkpoint_paths = {
        "vilt_curriculum": "vilt_models/checkpoint_curriculum_best.pt",
        "vilt_baseline":   "vilt_models/checkpoint_baseline_best.pt",
        "vilt":            "vilt_models/checkpoint_best.pt",
    }
    
    path = checkpoint_paths.get(model_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Checkpoint not available")

    return FileResponse(
        path=path,
        filename=f"{model_id}_checkpoint.pt",
        media_type="application/octet-stream",
    )


# ── Inference ─────────────────────────────────────────────────────────────────

@router.post("/inference/predict", response_model=VQAResponse)
async def predict(query: VQAQuery):
    return inference_service.predict(query)


# ── Upload ────────────────────────────────────────────────────────────────────

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {"filename": file.filename, "image_path": file_path}
