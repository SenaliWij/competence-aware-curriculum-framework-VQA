import os
from fastapi import HTTPException

class ModelService:
    def __init__(self):
        self.checkpoint_paths = {
            "vilt_curriculum": "vilt_models/checkpoint_best.pt",
            "vilt_baseline": "vilt_models/baseline_best.pt",
        }

    def get_model_download_path(self, model_id: str) -> str:
        """
        Retrieves the valid path for downloading a model checkpoint.
        Raises a 404 HTTPException if the model ID is invalid or the file is missing.
        """
        path = self.checkpoint_paths.get(model_id)
        if not path or not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Checkpoint not available")
        return path

model_service = ModelService()
