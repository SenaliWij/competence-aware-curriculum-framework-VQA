# checkpoint_service.py
"""
Centralised S3 checkpoint manager for the curriculum Training Stratergy.

Saves three checkpoint slots to S3:
    checkpoint_latest.pt   - overwritten every save
    checkpoint_best.pt     - only when validation accuracy improves
    checkpoint_step_N.pt  
"""
import io
import logging
import json
from typing import Dict, List, Optional, Any

import torch
import boto3

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Handles serialisation & AWS S3 uploads/downloads for all components.
    """

    def __init__(
        self,
        bucket: str,
        run_name: str,
        prefix: str = "checkpoints",
    ):
        self.bucket = bucket
        self.run_name = run_name
        self.prefix = prefix.rstrip("/")
        self.s3 = boto3.client("s3") # Boto3 is the standard Python library for interacting with AWS

        self.best_accuracy = 0.0

    # S3 helpers to manage cloud operations
    def _s3_key(self, name: str) -> str:
        """Build the full S3 object key."""
        return f"{self.prefix}/{self.run_name}/{name}"

    def _upload(self, state: Dict, key: str) -> None:
        """Serialise a state dict & upload to S3."""
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=buffer)

    def _download(self, key: str) -> Optional[Dict]:
        """Download & deserialise a state dict from S3"""
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            buffer = io.BytesIO(obj["Body"].read())
            return torch.load(buffer, map_location="cpu")
        except self.s3.exceptions.NoSuchKey:
            return None

    # State Collection
    @staticmethod
    def _collect_state(
        step: int,
        model,
        tracker,
        spl_sampler,
        eval_service,
        history: Optional[List[Dict]] = None,
        extra: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Gather every component's state into one dict.

        Parameters:
            step : Current training step.
            model : ModelAdapter - provides model + optimiser state dicts.
            tracker : CompetenceTracker - EMA entropy/loss values.
            spl_sampler : SoftSelfPacedSampler - per-sample loss records.
            eval_service: EvaluationService - validation metric history.
            history : Per-step training metrics log (optional).
            extra : Any additional metadata to store (optional).
        """
        state = {
            # Training progress
            "step": step,

            # Model weights + optimiser
            "model_state_dict": model.get_state_dict(),
            "optimizer_state_dict": model.get_optimizer_state_dict(),

            # Competence tracker (EMA values)
            "tracker": {
                "h_ema": tracker.h_ema,
                "l_ema": tracker.l_ema,
                "step_count": tracker.step_count,
            },

            # Soft self-paced sampler (per-sample losses)
            "spl_sampler": {
                "sample_losses": spl_sampler._sample_losses,
            },

            # Evaluation history
            "eval_history": eval_service.history,
            "training_history": history if history is not None else [],
        }

        if extra:
            state["extra"] = extra

        return state

    # Save curriculum model
    def save(
        self,
        step: int,
        model,
        tracker,
        spl_sampler,
        eval_service,
        accuracy: float = 0.0,
        save_numbered: bool = False,
        history: Optional[List[Dict]] = None,
        extra: Optional[Dict] = None,
    ) -> None:
        """
        Executes the saving process.
        
        1. Uploads 'latest'.
        2. If accuracy is a record-breaker, uploads 'best'.
        3. If requested, uploads a backup.
        4. Saves raw JSON logs

        """
        state = self._collect_state(
            step, model, tracker, spl_sampler, eval_service, history, extra
        )
        state["best_accuracy"] = self.best_accuracy

        # latest
        latest_key = self._s3_key("checkpoint_latest.pt")
        self._upload(state, latest_key)
        logger.info(
            "Saved latest checkpoint -> s3://%s/%s  (step %d)",
            self.bucket, latest_key, step,
        )

        # best
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            state["best_accuracy"] = self.best_accuracy
            best_key = self._s3_key("checkpoint_best.pt")
            self._upload(state, best_key)
            logger.info(
                "NEW BEST accuracy %.4f -> s3://%s/%s",
                accuracy, self.bucket, best_key,
            )

        # numbered 
        if save_numbered:
            num_key = self._s3_key(f"checkpoint_step_{step}.pt")
            self._upload(state, num_key)
            logger.info(
                "Saved numbered checkpoint -> s3://%s/%s",
                self.bucket, num_key,
            )

        # export logs as JSON
        if history is not None:
            try:
                log_key = self._s3_key("training_logs.json")
                self.s3.put_object(Bucket=self.bucket, Key=log_key, Body=json.dumps(history, indent=2))
            except Exception as e:
                logger.warning(f"Failed to export training_logs.json to S3: {e}")
                
        if eval_service.history:
            try:
                eval_key = self._s3_key("eval_logs.json")
                self.s3.put_object(Bucket=self.bucket, Key=eval_key, Body=json.dumps(eval_service.history, indent=2))
            except Exception as e:
                logger.warning(f"Failed to export eval_logs.json to S3: {e}")

    # Save baseline model
    def save_baseline(
        self,
        step: int,
        model,
        accuracy: float = 0.0,
        save_numbered: bool = False,
        extra: Optional[Dict] = None,
    ) -> None:
        """
        Save a baseline model checkpoint to S3 (no tracker, sampler).
        """
        state = {
            "step": step,
            "model_state_dict": model.get_state_dict(),
            "optimizer_state_dict": model.get_optimizer_state_dict(),
        }
        if extra:
            state["extra"] = extra

        state["best_accuracy"] = self.best_accuracy

        # latest
        latest_key = self._s3_key("baseline_checkpoint_latest.pt")
        self._upload(state, latest_key)
        logger.info(
            "Saved latest baseline checkpoint -> s3://%s/%s  (step %d)",
            self.bucket, latest_key, step,
        )

        # best
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            state["best_accuracy"] = self.best_accuracy
            best_key = self._s3_key("baseline_checkpoint_best.pt")
            self._upload(state, best_key)
            logger.info(
                "NEW BEST baseline accuracy %.4f -> s3://%s/%s",
                accuracy, self.bucket, best_key,
            )

        # numbered
        if save_numbered:
            num_key = self._s3_key(f"baseline_checkpoint_step_{step}.pt")
            self._upload(state, num_key)
            logger.info(
                "Saved numbered baseline checkpoint -> s3://%s/%s",
                self.bucket, num_key,
            )

    # Load baseline model
    def load_baseline(
        self,
        model,
        tag: str = "latest",
    ) -> Optional[int]:
        """
        Download a baseline checkpoint from S3 & restore model state.

        Parameters
            model: ModelAdapter whose weights will be restored.
            tag  : Which slot to load: latest or best.

        Returns:
            Training step the checkpoint was saved at, or None if not found.
        """
        filename = f"baseline_checkpoint_{tag}.pt"
        key = self._s3_key(filename)

        state = self._download(key)
        if state is None:
            logger.info("No baseline checkpoint found at s3://%s/%s.", self.bucket, key)
            return None

        logger.info("Loaded baseline checkpoint from s3://%s/%s", self.bucket, key)

        model.load(state["model_state_dict"])
        if hasattr(model, "load_optimizer_state") & "optimizer_state_dict" in state:
            model.load_optimizer_state(state["optimizer_state_dict"])

        self.best_accuracy = state.get("best_accuracy", 0.0)
        step = state.get("step", 0)

        logger.info(
            "Resumed baseline at step %d | best_accuracy=%.4f",
            step, self.best_accuracy,
        )
        return step

    # Load  curriculum trained model
    def load(
        self,
        model,
        tracker,
        spl_sampler,
        eval_service,
        tag: str = "latest",
    ) -> Optional[int]:
        """
        Download a checkpoint from S3 & restore all component states.
        
        Parameters:
            model: ModelAdapter whose weights will be restored.
            tracker: CompetenceTracker whose EMA state will be restored.
            spl_sampler: SoftSelfPacedSampler whose loss records will be restored.
            eval_service: EvaluationService whose history will be restored.
            tag: Which slot to load - latest or best.

        Returns:
            Tuple of (step, history) - the training step & metrics log at
            the time of saving.
        """
        filename = f"checkpoint_{tag}.pt"
        key = self._s3_key(filename)

        state = self._download(key)
        if state is None:
            logger.info("No checkpoint found at s3://%s/%s- starting fresh.", self.bucket, key)
            return None, []

        logger.info("Loaded checkpoint from s3://%s/%s", self.bucket, key)

        # Model
        model.load(state["model_state_dict"])
        if hasattr(model, "load_optimizer_state") & "optimizer_state_dict" in state:
            model.load_optimizer_state(state["optimizer_state_dict"])

        # Competence tracker
        t = state.get("tracker", {})
        tracker.h_ema = t.get("h_ema", tracker.h_ema)
        tracker.l_ema = t.get("l_ema", tracker.l_ema)
        tracker.step_count = t.get("step_count", tracker.step_count)

        # SPL sampler
        spl = state.get("spl_sampler", {})
        spl_sampler._sample_losses = spl.get("sample_losses", {})

        # Evaluation history
        eval_service.history = state.get("eval_history", [])

        # Managers own tracked state
        self.best_accuracy = state.get("best_accuracy", 0.0)

        step = state.get("step", 0)
        history = state.get("training_history", [])
        
        logger.info(
            "Resumed at step %d | best_accuracy=%.4f | C=%.3f",
            step, self.best_accuracy, tracker.competence(),
        )
        return step, history
