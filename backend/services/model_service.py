"""
Model registry – stores metadata about pre-trained models.
No real-time training; models are trained offline and registered here.
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
        training_type="curriculum",
        dataset="CLEVR (5 tiers by reasoning complexity)",
        total_steps=10000,
        tier_accuracies={
            "Tier 1 — Attribute & existence": 95.2,
            "Tier 2 — Compare attribute": 97.4,
            "Tier 3 — Counting / integer compare": 94.9,
            "Tier 4 — Attribute with relate": 88.4,
            "Tier 5 — Full composition": 84.3,
        },
        config={
            "base_model": "dandelin/vilt-b32-finetuned-vqa",
            "optimizer": "AdamW",
            "learning_rate": 5e-5,
            "batch_size": 32,
            "ema_beta": 0.9,
            "entropy_weight": 0.7,
            "tier_difficulties": [1.0, 3.0, 5.0, 7.0, 10.0],
            "spl_lambda_init": 0.5,
            "spl_lambda_max": 5.0,
        },
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
        training_type="baseline",
        dataset="CLEVR (all tiers shuffled)",
        total_steps=10000,
        tier_accuracies={
            "Tier 1 — Attribute & existence": 92.1,
            "Tier 2 — Compare attribute": 90.5,
            "Tier 3 — Counting / integer compare": 85.3,
            "Tier 4 — Attribute with relate": 78.6,
            "Tier 5 — Full composition": 72.4,
        },
        config={
            "base_model": "dandelin/vilt-b32-finetuned-vqa",
            "optimizer": "AdamW",
            "learning_rate": 5e-5,
            "batch_size": 32,
        },
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
