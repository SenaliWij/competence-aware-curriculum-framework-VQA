import os
import json
import torch
import boto3
from PIL import Image
from transformers import ViltProcessor, ViltForQuestionAnswering
from models.schemas import VQAQuery, VQAResponse


# CONFIG
S3_BUCKET = "clevr-dataset"
S3_CHECKPOINT_KEY = "vqa_checkpoints/curriculum_run_v1/checkpoint_latest.pt"
S3_VOCAB_KEY = "vqa_checkpoints/answer_vocab.json"
LOCAL_MODEL_PATH = "vilt_models/checkpoint_tier2_best.pt"

# INFERENCE SERVICE

class InferenceService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.s3 = boto3.client("s3")

        # Load processor (same as training)
        self.processor = ViltProcessor.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa",
            use_fast=False
        )

        #  Load vocab from S3
        self.id2label, self.label2id = self._load_answer_vocab_from_s3()

        # Load model from S3 checkpoint
        self.model = self._load_model()

    # DOWNLOAD CHECKPOINT FROM S3
    def _download_checkpoint(self):
        if os.path.exists(LOCAL_MODEL_PATH):
            return

        os.makedirs(os.path.dirname(LOCAL_MODEL_PATH), exist_ok=True)

        print(f"Downloading checkpoint from s3://{S3_BUCKET}/{S3_CHECKPOINT_KEY}")
        self.s3.download_file(S3_BUCKET, S3_CHECKPOINT_KEY, LOCAL_MODEL_PATH)
        print("Checkpoint download complete")

    # LOAD ANSWER VOCAB FROM S3
    def _load_answer_vocab_from_s3(self):
        print(f"Loading answer vocab from s3://{S3_BUCKET}/{S3_VOCAB_KEY}")

        obj = self.s3.get_object(
            Bucket=S3_BUCKET,
            Key=S3_VOCAB_KEY
        )

        vocab = json.loads(obj["Body"].read().decode("utf-8"))

        label2id = {k: int(v) for k, v in vocab.items()}
        id2label = {int(v): k for k, v in vocab.items()}

        print("Loaded answer vocab with", len(id2label), "labels")

        return id2label, label2id

    # LOAD MODEL + CHECKPOINT
    def _load_model(self):
        self._download_checkpoint()

        num_labels = len(self.id2label)

        # Initialize architecture exactly like training
        model = ViltForQuestionAnswering.from_pretrained(
            "dandelin/vilt-b32-finetuned-vqa",
            num_labels=num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True
        )

        checkpoint = torch.load(LOCAL_MODEL_PATH, map_location=self.device)

        if "model_state_dict" not in checkpoint:
            raise RuntimeError("Checkpoint missing 'model_state_dict'")

        state_dict = checkpoint["model_state_dict"]

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        print("Missing keys:", missing)
        print("Unexpected keys:", unexpected)

        model.to(self.device)
        model.eval()

        print("Model loaded successfully (Tier 2 Best)")

        return model

    # PREDICTION
    @torch.no_grad()
    def predict(self, query: VQAQuery) -> VQAResponse:
        if not query.image_path:
            raise ValueError("image_path is required")

        if not os.path.exists(query.image_path):
            raise FileNotFoundError(
                f"Image not found at {query.image_path}"
            )

        image = Image.open(query.image_path).convert("RGB")

        encoding = self.processor(
            images=image,
            text=query.question,
            return_tensors="pt"
        )

        encoding = {k: v.to(self.device) for k, v in encoding.items()}

        outputs = self.model(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
            pixel_values=encoding["pixel_values"]
        )

        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

        top_prob, top_idx = probs.max(dim=-1)

        answer = self.id2label[top_idx.item()]
        confidence = round(top_prob.item(), 3)

        # Top 3 predictions
        topk = torch.topk(probs, k=min(3, self.model.config.num_labels))

        candidates = [
            {
                "text": self.id2label[idx.item()],
                "confidence": round(prob.item(), 3)
            }
            for prob, idx in zip(topk.values[0], topk.indices[0])
        ]

        reasoning = [
            "Image encoded using ViLT vision transformer",
            "Question encoded using text transformer",
            "Multimodal fusion performed",
            "Answer predicted using curriculum-trained classifier (Tier 2 best)"
        ]

        return VQAResponse(
            answer=answer,
            confidence=confidence,
            candidate_answers=candidates,
            reasoning_trace=reasoning
        )


# Initialize service
inference_service = InferenceService()
