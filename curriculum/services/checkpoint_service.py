# checkpoint_service_s3.py

import torch
import boto3
import io
from typing import Dict, Optional

class CheckpointManager:
    """
    Manages checkpoints directly in S3 (NO local disk usage).
    Supports:
    - latest checkpoint
    - best per tier
    - best overall
    - resume training
    """

    def __init__(
        self,
        bucket: str,
        run_name: str,
        prefix: str = "checkpoints"
    ):

        self.bucket = bucket
        self.run_name = run_name
        self.prefix = prefix.rstrip("/")
        self.s3 = boto3.client("s3")

        self.best_overall_acc = 0.0

    # helpers to upload and download from S3 checkpoints
    def _s3_key(self, name: str) -> str:
        return f"{self.prefix}/{self.run_name}/{name}"

    def _upload_state(self, state: Dict, key: str):
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)

        self.s3.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=buffer
        )

    def _download_state(self, key: str) -> Optional[Dict]:
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            buffer = io.BytesIO(obj["Body"].read())
            return torch.load(buffer, map_location="cpu")
        except self.s3.exceptions.NoSuchKey:
            return None


    def save(
        self,
        model_state: Dict,
        optimizer_state: Dict,
        curriculum_state: Dict,
        metrics: Dict,
        tier: int,
        is_best: bool = False
    ):
        state = {
            "model_state_dict": model_state,
            "optimizer_state_dict": optimizer_state,
            "curriculum_state": curriculum_state,
            "metrics": metrics,
            "tier": tier,
            "best_overall_acc": self.best_overall_acc
        }

        # Save latest 
        latest_key = self._s3_key("checkpoint_latest.pt")
        self._upload_state(state, latest_key)

        # Tier best
        if is_best:
            tier_best_key = self._s3_key(f"checkpoint_tier{tier}_best.pt")
            self._upload_state(state, tier_best_key)
            print(f"Saved BEST model for Tier {tier} → s3://{self.bucket}/{tier_best_key}")

        #  Overall best
        current_acc = metrics.get("accuracy", 0.0)
        if current_acc > self.best_overall_acc:
            self.best_overall_acc = current_acc
            state["best_overall_acc"] = self.best_overall_acc

            overall_key = self._s3_key("checkpoint_best_overall.pt")
            self._upload_state(state, overall_key)

            # Also update latest with new best value
            self._upload_state(state, latest_key)

            print(
                f"NEW OVERALL BEST: {self.best_overall_acc:.4f} "
                f"→ s3://{self.bucket}/{overall_key}"
            )

    def load_latest(self, model, optimizer=None, curriculum=None):
        latest_key = self._s3_key("checkpoint_latest.pt")

        checkpoint = self._download_state(latest_key)
        if checkpoint is None:
            print("No checkpoint found in S3. Starting from scratch.")
            return 1, None

        print(f" Loaded checkpoint from s3://{self.bucket}/{latest_key}")

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if curriculum and "curriculum_state" in checkpoint:
            curriculum.load_config_state(checkpoint["curriculum_state"])

        self.best_overall_acc = checkpoint.get("best_overall_acc", 0.0)

        tier = checkpoint.get("tier", 1)
        metrics = checkpoint.get("metrics", None)

        print(
            f" Resumed at Tier {tier} | "
            f"Best Overall Acc: {self.best_overall_acc:.4f}"
        )

        return tier, metrics
