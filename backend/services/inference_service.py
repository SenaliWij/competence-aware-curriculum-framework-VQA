import os
import json
import logging
import torch
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
from models.schemas import VQAQuery, VQAResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

#Constants
BASE_MODEL_ID   = "dandelin/vilt-b32-finetuned-vqa"
DEFAULT_MODEL   = "vilt_curriculum"
TOP_K_ANSWERS   = 3

MODEL_CHECKPOINT_PATHS: dict[str, str] = {
    "vilt_curriculum": os.path.join("vilt_models", "checkpoint_best.pt"),
    "vilt_baseline":   os.path.join("vilt_models", "baseline_best.pt"),
}

VOCAB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "answer_vocab.json")
)


class InferenceService:
    """Loads ViLT checkpoints and runs VQA inference."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("InferenceService initializing on device: %s", self.device)

        self.processor  = self._load_processor()
        self.id2label, self.label2id = self._load_answer_vocab()
        self._model_cache: dict[str, ViltForQuestionAnswering] = {}

        logger.info(
            "InferenceService ready — %d answer labels, %d checkpoints available",
            len(self.id2label), len(MODEL_CHECKPOINT_PATHS),
        )

    #Helpers
    def _load_processor(self) -> ViltProcessor:
        logger.info("Loading ViLT processor from '%s'", BASE_MODEL_ID)
        processor = ViltProcessor.from_pretrained(BASE_MODEL_ID, use_fast=False)
        logger.info("Processor loaded")
        return processor

    def _load_answer_vocab(self) -> tuple[dict[int, str], dict[str, int]]:
        logger.info("Loading answer vocab from '%s'", VOCAB_PATH)

        if not os.path.exists(VOCAB_PATH):
            raise FileNotFoundError(f"Answer vocab not found: {VOCAB_PATH}")

        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab: dict[str, str] = json.load(f)

        label2id = {k: int(v) for k, v in vocab.items()}
        id2label = {int(v): k for k, v in vocab.items()}

        logger.info("Vocab loaded — %d labels", len(id2label))
        return id2label, label2id

    def _resolve_model_id(self, model_id: str | None) -> str:
        """Return a valid model_id, falling back to default if unknown."""
        if model_id in MODEL_CHECKPOINT_PATHS:
            return model_id
        logger.warning(
            "Unknown model_id '%s' — falling back to '%s'", model_id, DEFAULT_MODEL
        )
        return DEFAULT_MODEL

    def _load_model(self, model_id: str) -> ViltForQuestionAnswering:
        if model_id in self._model_cache:
            logger.debug("Cache hit for model '%s'", model_id)
            return self._model_cache[model_id]

        checkpoint_path = MODEL_CHECKPOINT_PATHS[model_id]
        logger.info("Loading checkpoint for '%s' from '%s'", model_id, checkpoint_path)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        num_labels = len(self.id2label)

        model = ViltForQuestionAnswering.from_pretrained(
            BASE_MODEL_ID,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        if "model_state_dict" not in checkpoint:
            raise RuntimeError(
                f"Checkpoint '{checkpoint_path}' is missing 'model_state_dict'"
            )

        state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(state_dict, strict=False)

        model_keys      = set(model.state_dict().keys())
        checkpoint_keys = set(state_dict.keys())
        missing     = model_keys - checkpoint_keys
        unexpected  = checkpoint_keys - model_keys

        if missing:
            logger.warning("Keys missing from checkpoint (%d): %s", len(missing), missing)
        if unexpected:
            logger.warning("Unexpected keys in checkpoint (%d): %s", len(unexpected), unexpected)

        head_weight = state_dict.get("classifier.out_proj.weight")
        if head_weight is not None:
            ckpt_labels = head_weight.shape[0]
            if ckpt_labels != num_labels:
                logger.warning(
                    "Label count mismatch — checkpoint: %d, vocab: %d",
                    ckpt_labels, num_labels,
                )
            else:
                logger.info("Label counts match: %d", num_labels)

        model.to(self.device)
        model.eval()

        self._model_cache[model_id] = model
        logger.info("Model '%s' loaded and cached", model_id)
        return model


    @torch.no_grad()
    def predict(self, query: VQAQuery) -> VQAResponse:
        model_id = self._resolve_model_id(query.model_id)
        logger.info(
            "predict() — model='%s', question='%s', image='%s'",
            model_id, query.question, query.image_path,
        )

        if not query.image_path:
            raise ValueError("'image_path' is required")
        if not os.path.exists(query.image_path):
            raise FileNotFoundError(f"Image not found: {query.image_path}")

        model = self._load_model(model_id)

        # Load the image, run inference, then delete the upload so files don't pile up.
        image = None
        try:
            image = Image.open(query.image_path).convert("RGB")
            encoding = self.processor(images=image, text=query.question, return_tensors="pt")
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            # Forward pass
            outputs = model(
                input_ids=encoding["input_ids"],
                attention_mask=encoding["attention_mask"],
                pixel_values=encoding["pixel_values"],
            )
        finally:
            # Close file handle first (Windows can't delete an open file).
            try:
                if image is not None:
                    image.close()
            except Exception:
                pass

            # Only delete files inside the uploads folder (basic safety check).
            try:
                uploads_dir = os.path.abspath("uploads")
                img_path = os.path.abspath(query.image_path)
                if os.path.commonpath([uploads_dir, img_path]) == uploads_dir and os.path.exists(img_path):
                    os.remove(img_path)
                    logger.info("Deleted uploaded file: %s", img_path)
            except Exception as e:
                logger.warning("Could not delete uploaded file '%s': %s", query.image_path, e)

        # Decode
        probs       = torch.softmax(outputs.logits, dim=-1)
        top_prob, top_idx = probs.max(dim=-1)

        answer      = self.id2label[top_idx.item()]
        confidence  = round(top_prob.item(), 4)

        topk = torch.topk(probs, k=min(TOP_K_ANSWERS, model.config.num_labels))
        candidates = [
            {"text": self.id2label[idx.item()], "confidence": round(prob.item(), 4)}
            for prob, idx in zip(topk.values[0], topk.indices[0])
        ]
        logger.info(
            "Prediction — answer='%s', confidence=%.4f, candidates=%s",
            answer, confidence, candidates,
        )

        return VQAResponse(
            answer=answer,
            confidence=confidence,
            candidate_answers=candidates,
        )

inference_service = InferenceService()