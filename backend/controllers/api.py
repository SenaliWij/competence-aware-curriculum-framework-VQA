import os
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from models.schemas import VQAQuery, VQAResponse, VQACompareQuery, VQACompareResponse
from services.inference_service import inference_service
from services.model_service import model_service

router = APIRouter()

# Download an enhanced VQA model checkpoint by model ID
@router.get("/models/{model_id}/download")
async def download_model(model_id: str):
    path = model_service.get_model_download_path(model_id)

    return FileResponse(
        path=path,
        filename=f"{model_id}_checkpoint.pt",
        media_type="application/octet-stream",
    )

# Run VQA inference on a single image-question pair
@router.post("/inference/predict", response_model=VQAResponse)
async def predict(query: VQAQuery):
    return inference_service.predict(query)

#Compare VQA predictions across multiple models for the same image-question pair
@router.post("/inference/compare", response_model=VQACompareResponse)
async def compare(query: VQACompareQuery):
    return inference_service.compare(query)

# Upload an image file to the server for use in subsequent inference requests
@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)

    file_path = os.path.join(upload_dir, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    return {"filename": file.filename, "image_path": file_path}

