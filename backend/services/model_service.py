"""
Model registry – stores metadata about pre-trained models.
 models are trained offline and registered here.
"""

from models.schemas import ModelInfo

MODELS = {
    "vilt_curriculum": ModelInfo(
        id="vilt_curriculum",
        name="ViLT — Curriculum-Trained",
        description=(
            "ViLT model trained with the competence-aware curriculum framework. "
            "Uses a hybrid competence function (entropy + loss), power-law tier "
            "sampling (macro), and soft self-paced learning (micro) over 10 000 steps on CLEVR."
        ),
        available=True,
    ),
    "vilt_baseline": ModelInfo(
        id="vilt_baseline",
        name="ViLT — Baseline (Flat Training)",
        description=(
            "ViLT model trained without curriculum learning. "
            "All tiers are mixed randomly from step 1 (flat training). "
            "Serves as a comparison baseline."
        ),
        available=True,
    ),
}


class ModelService:
    """Read-only service that returns model metadata."""

    def list_models(self):
        return list(MODELS.values())

    def get_model(self, model_id: str):
        return MODELS.get(model_id)


model_service = ModelService()
