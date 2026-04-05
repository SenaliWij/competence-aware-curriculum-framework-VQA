import os
import json
import logging
import torch
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
from models.schemas import (
    VQAQuery, VQAResponse,
    VQACompareQuery, VQACompareResponse, VQAModelResult,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants
BASE_MODEL_ID = "dandelin/vilt-b32-finetuned-vqa"
DEFAULT_MODEL = "vilt_curriculum"
TOP_K_ANSWERS = 3 # number of top candidate answers

MODEL_CHECKPOINT_PATHS: dict[str, str] = {
    "vilt_curriculum": os.path.join("vilt_models", "checkpoint_best.pt"),
    "vilt_baseline":   os.path.join("vilt_models", "baseline_best.pt"),
}

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "vilt_curriculum": "VILT-CL",
    "vilt_baseline":   "VILT-Baseline",
}

VOCAB_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "answer_vocab.json")
)

class InferenceService:
    """Loads ViLT checkpoints & runs VQA inference."""

    def __init__(self):
        # Setup computation device and fallback to CPU if no GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("InferenceService initializing on device: %s", self.device)

        # Load the processor and answer vocab
        self.processor = self._load_processor()
        self.id2label, self.label2id = self._load_answer_vocab()
        self._model_cache: dict[str, ViltForQuestionAnswering] = {}

        logger.info(
            "InferenceService ready - %d answer labels, %d checkpoints available",
            len(self.id2label), len(MODEL_CHECKPOINT_PATHS),
        )

    # Helper functions

    def _load_processor(self) -> ViltProcessor:
        """Download or load from cache the ViLT image-text processor."""
        logger.info("Loading ViLT processor from '%s'", BASE_MODEL_ID)
        processor = ViltProcessor.from_pretrained(BASE_MODEL_ID, use_fast=False)
        logger.info("Processor loaded")
        return processor

    def _load_answer_vocab(self) -> tuple[dict[int, str], dict[str, int]]:
        """
        Parse answer_vocab.json into two lookup dictionaries.

        Returns:
            id2label: Maps integer label IDs -> answer strings.
            label2id: Maps answer strings -> integer label IDs.
        """
        logger.info("Loading answer vocab from '%s'", VOCAB_PATH)
        if not os.path.exists(VOCAB_PATH):
            raise FileNotFoundError(f"Answer vocab not found: {VOCAB_PATH}")

        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab: dict[str, str] = json.load(f)

        label2id = {k: int(v) for k, v in vocab.items()}
        id2label = {int(v): k for k, v in vocab.items()}
        logger.info("Vocab loaded - %d labels", len(id2label))
        return id2label, label2id

    def _load_model(self, model_id: str) -> ViltForQuestionAnswering:
        """
        Load a ViLT model for the given model_id checkpoint into memory and cache it.

        Returns:
            A ViLT model in eval mode, moved to the correct device.
        """
        if model_id in self._model_cache:
            logger.debug("Cache hit for model '%s'", model_id)
            return self._model_cache[model_id]

        checkpoint_path = MODEL_CHECKPOINT_PATHS[model_id]
        logger.info("Loading checkpoint for '%s' from '%s'", model_id, checkpoint_path)
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        num_labels = len(self.id2label)

        # Initialise the architecture from HuggingFace with our custom label set.
        model = ViltForQuestionAnswering.from_pretrained(
            BASE_MODEL_ID,
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )
        
        # Load the saved weights, map_location ensures GPU checkpoints load on CPU too
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        if "model_state_dict" not in checkpoint:
            raise RuntimeError(
                f"Checkpoint '{checkpoint_path}' is missing 'model_state_dict'"
            )

        state_dict = checkpoint["model_state_dict"]
        model.load_state_dict(state_dict, strict=False) # strict=False allows partial loading

        # Log any weight mismatches to help diagnose checkpoint compatibility issues
        missing = set(model.state_dict().keys()) - set(state_dict.keys())
        unexpected = set(state_dict.keys()) - set(model.state_dict().keys())
        if missing:
            logger.warning("Keys missing from checkpoint (%d): %s", len(missing), missing)
        if unexpected:
            logger.warning("Unexpected keys in checkpoint (%d): %s", len(unexpected), unexpected)

        # Move model to GPU/CPU and set to evaluation mode
        model.to(self.device)
        model.eval()
        self._model_cache[model_id] = model
        logger.info("Model '%s' loaded & cached", model_id)
        return model

    def _encode_image(self, image_path: str, question: str) -> dict:
        """
        Load an image from disk, encode it with the question, then delete
        the source file to prevent accumulation of uploaded images.

        Parameters:
            image_path: Path to the uploaded image file.
            question: question to pair with the image.

        Returns:
            A dict of tensors already moved to the correct device."""
        if not image_path:
            raise ValueError("'image_path' is required")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = None
        try:
            # Convert to RGB to ensure consistency
            image = Image.open(image_path).convert("RGB")
            
            # Encode image and question
            encoding = self.processor(images=image, text=question, return_tensors="pt")
            
            # Move tensors to the correct device
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
            return encoding
        finally:
            # Close file handle (Windows cannot delete an open file).
            try:
                if image is not None:
                    image.close()
            except Exception:
                pass
            # Delete uploaded file so they don't pile up.
            self._cleanup_upload(image_path)

    def _cleanup_upload(self, image_path: str):
        """Remove the file only if it lives inside the uploads/ directory."""
        try:
            uploads_dir = os.path.abspath("uploads")
            abs_path = os.path.abspath(image_path)
            if os.path.commonpath([uploads_dir, abs_path]) == uploads_dir and os.path.exists(abs_path):
                os.remove(abs_path)
                logger.info("Deleted uploaded file: %s", abs_path)
        except Exception as e:
            logger.warning("Could not delete uploaded file '%s': %s", image_path, e)

    def _run_single_model(self, model_id: str, encoding: dict) -> dict:
        """
        Execute a forward pass for one model and decode the top-k predictions.

        Parameters:
            model_id: Which checkpoint to run.
            encoding: Pre-computed tensor dict from _encode_image().

        Returns:
            A dict containing the correct answer, top answers and their confidence scores.
        """
        model = self._load_model(model_id)

        # Forward pass 
        outputs = model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            pixel_values=encoding["pixel_values"],
        )

        # Convert raw logits to a probability distribution
        probs = torch.softmax(outputs.logits, dim=-1)
        # Extract the single highest-confidence answer
        top_prob, top_idx = probs.max(dim=-1)

        answer = self.id2label[top_idx.item()]
        confidence = round(top_prob.item(), 4)
        
        # Build the ranked candidate list
        topk = torch.topk(probs, k=min(TOP_K_ANSWERS, model.config.num_labels))
        candidates = [
            {"text": self.id2label[idx.item()], "confidence": round(prob.item(), 4)}
            for prob, idx in zip(topk.values[0], topk.indices[0])
        ]

        return {
            "model_id": model_id,
            "model_name": MODEL_DISPLAY_NAMES.get(model_id, model_id),
            "answer": answer,
            "confidence": confidence,
            "candidate_answers": candidates,
        }


    @torch.no_grad()
    def predict(self, query: VQAQuery) -> VQAResponse:
        """
        Run single-model VQA inference using the default curriculum-trained model.

        Parameters:
            query: Contains image_path and the question.

        Returns:
            VQAResponse.
        """
        model_id = DEFAULT_MODEL
        logger.info(
            "predict() - model='%s', question='%s', image='%s'",
            model_id, query.question, query.image_path,
        )
        
        # Encode once, run inference, decode result
        encoding = self._encode_image(query.image_path, query.question)
        result = self._run_single_model(model_id, encoding)

        logger.info(
            "Prediction - answer='%s', confidence=%.4f",
            result["answer"], result["confidence"],
        )
        return VQAResponse(
            answer=result["answer"],
            confidence=result["confidence"],
            candidate_answers=result["candidate_answers"],
        )

    @torch.no_grad()
    def compare(self, query: VQACompareQuery) -> VQACompareResponse:
        """
        Run both models on the same image + question and return results side by side.

        The image is encoded once and the same tensor dict is reused for both
        forward passes.

        Parameters:
            query: Contains image_path and the question.

        Returns:
            VQACompareResponse.
        """
        logger.info(
            "compare() - question='%s', image='%s'",
            query.question, query.image_path,
        )
        # Encode the image once and share the tensors across both model runs
        encoding = self._encode_image(query.image_path, query.question)
        proposed = self._run_single_model("vilt_curriculum", encoding)
        baseline = self._run_single_model("vilt_baseline", encoding)

        logger.info(
            "Comparison complete - proposed='%s' (%.4f), baseline='%s' (%.4f)",
            proposed["answer"], proposed["confidence"],
            baseline["answer"], baseline["confidence"],
        )
        return VQACompareResponse(
            proposed=VQAModelResult(**proposed),
            baseline=VQAModelResult(**baseline),
        )


inference_service = InferenceService()