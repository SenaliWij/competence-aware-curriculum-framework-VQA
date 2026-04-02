# checkpoint_service.py
"""
Centralised S3 checkpoint manager for the curriculum framework.

Every component's state is collected and saved/restored through
this single service — no other file handles its own serialisation.

Saves three checkpoint slots to S3:
  • checkpoint_latest.pt    — overwritten every save
  • checkpoint_best.pt      — only when validation accuracy improves
  • checkpoint_step_N.pt    — periodic numbered snapshots (optional)
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
    Collects state from every curriculum component, serialises it,
    and uploads / downloads from S3.

    Parameters
    ----------
    bucket : str
        S3 bucket name.
    run_name : str
        Unique identifier for this training run (used in the S3 key path).
    prefix : str
        S3 key prefix for checkpoints.
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
        self.s3 = boto3.client("s3")

        self.best_accuracy = 0.0

    # ==================================================================
    # S3 helpers
    # ==================================================================
    def _s3_key(self, name: str) -> str:
        """Build the full S3 object key."""
        return f"{self.prefix}/{self.run_name}/{name}"

    def _upload(self, state: Dict, key: str) -> None:
        """Serialise a state dict and upload to S3."""
        buffer = io.BytesIO()
        torch.save(state, buffer)
        buffer.seek(0)
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=buffer)

    def _download(self, key: str) -> Optional[Dict]:
        """Download and deserialise a state dict from S3 (None if missing)."""
        try:
            obj = self.s3.get_object(Bucket=self.bucket, Key=key)
            buffer = io.BytesIO(obj["Body"].read())
            return torch.load(buffer, map_location="cpu")
        except self.s3.exceptions.NoSuchKey:
            return None

    # ==================================================================
    # Collect state from all components
    # ==================================================================
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

        Parameters
        ----------
        step : int
            Current training step.
        model : ModelAdapter
            The VQA model (provides state_dict + optimizer state).
        tracker : CompetenceTracker
            EMA entropy/loss tracker.
        spl_sampler : SoftSelfPacedSampler
            Per-sample loss records.
        eval_service : EvaluationService
            Validation metric history.
        history : list of dict | None
            Log of per-step training metrics.
        extra : dict | None
            Any additional metadata to include.
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

    # ==================================================================
    # Save
    # ==================================================================
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
        Save a checkpoint to S3.

        Always saves ``checkpoint_latest.pt``.
        If ``accuracy`` beats the previous best, also saves ``checkpoint_best.pt``.
        If ``save_numbered`` is True, saves ``checkpoint_step_{step}.pt``.

        Parameters
        ----------
        step : int
            Current training step.
        model : ModelAdapter
        tracker : CompetenceTracker
        spl_sampler : SoftSelfPacedSampler
        eval_service : EvaluationService
        accuracy : float
            Current validation accuracy (for best-model tracking).
        save_numbered : bool
            Whether to also save a step-numbered snapshot.
        history : list of dict | None
            Log of per-step training metrics.
        extra : dict | None
            Additional metadata to store.
        """
        state = self._collect_state(
            step, model, tracker, spl_sampler, eval_service, history, extra
        )
        state["best_accuracy"] = self.best_accuracy

        # ---- Latest ----
        latest_key = self._s3_key("checkpoint_latest.pt")
        self._upload(state, latest_key)
        logger.info(
            "Saved latest checkpoint → s3://%s/%s  (step %d)",
            self.bucket, latest_key, step,
        )

        # ---- Best ----
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            state["best_accuracy"] = self.best_accuracy
            best_key = self._s3_key("checkpoint_best.pt")
            self._upload(state, best_key)
            logger.info(
                "NEW BEST accuracy %.4f → s3://%s/%s",
                accuracy, self.bucket, best_key,
            )

        # ---- Numbered (optional) ----
        if save_numbered:
            num_key = self._s3_key(f"checkpoint_step_{step}.pt")
            self._upload(state, num_key)
            logger.info(
                "Saved numbered checkpoint → s3://%s/%s",
                self.bucket, num_key,
            )

        # ---- Export logs as JSON to S3 ----
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

    def save_baseline(
        self,
        step: int,
        model,
        accuracy: float = 0.0,
        save_numbered: bool = False,
        extra: Optional[Dict] = None,
    ) -> None:
        """
        Save a baseline model checkpoint to S3 (no tracker, sampler, etc.).
        """
        state = {
            "step": step,
            "model_state_dict": model.get_state_dict(),
            "optimizer_state_dict": model.get_optimizer_state_dict(),
        }
        if extra:
            state["extra"] = extra

        state["best_accuracy"] = self.best_accuracy

        # ---- Latest ----
        latest_key = self._s3_key("baseline_checkpoint_latest.pt")
        self._upload(state, latest_key)
        logger.info(
            "Saved latest baseline checkpoint → s3://%s/%s  (step %d)",
            self.bucket, latest_key, step,
        )

        # ---- Best ----
        if accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
            state["best_accuracy"] = self.best_accuracy
            best_key = self._s3_key("baseline_checkpoint_best.pt")
            self._upload(state, best_key)
            logger.info(
                "NEW BEST baseline accuracy %.4f → s3://%s/%s",
                accuracy, self.bucket, best_key,
            )

        # ---- Numbered (optional) ----
        if save_numbered:
            num_key = self._s3_key(f"baseline_checkpoint_step_{step}.pt")
            self._upload(state, num_key)
            logger.info(
                "Saved numbered baseline checkpoint → s3://%s/%s",
                self.bucket, num_key,
            )

    def load_baseline(
        self,
        model,
        tag: str = "latest",
    ) -> Optional[int]:
        """
        Download a baseline checkpoint from S3 and restore model state.

        Parameters
        ----------
        model : ModelAdapter
        tag : str
            Which checkpoint to load: ``"latest"`` or ``"best"``.

        Returns
        -------
        int | None
            The training step the checkpoint was saved at, or None if no checkpoint.
        """
        filename = f"baseline_checkpoint_{tag}.pt"
        key = self._s3_key(filename)

        state = self._download(key)
        if state is None:
            logger.info("No baseline checkpoint found at s3://%s/%s.", self.bucket, key)
            return None

        logger.info("Loaded baseline checkpoint from s3://%s/%s", self.bucket, key)

        model.load(state["model_state_dict"])
        if hasattr(model, "load_optimizer_state") and "optimizer_state_dict" in state:
            model.load_optimizer_state(state["optimizer_state_dict"])

        self.best_accuracy = state.get("best_accuracy", 0.0)
        step = state.get("step", 0)

        logger.info(
            "Resumed baseline at step %d | best_accuracy=%.4f",
            step, self.best_accuracy,
        )
        return step

    # ==================================================================
    # Load
    # ==================================================================
    def load(
        self,
        model,
        tracker,
        spl_sampler,
        eval_service,
        tag: str = "latest",
    ) -> Optional[int]:
        """
        Download a checkpoint from S3 and restore all component states.

        Parameters
        ----------
        model : ModelAdapter
        tracker : CompetenceTracker
        spl_sampler : SoftSelfPacedSampler
        eval_service : EvaluationService
        tag : str
            Which checkpoint to load: ``"latest"`` or ``"best"``.

        Returns
        -------
        tuple[int | None, list[dict]]
            A tuple of (step, history) where step is the training step
            the checkpoint was saved at, or (None, []) if no checkpoint.
        """
        filename = f"checkpoint_{tag}.pt"
        key = self._s3_key(filename)

        state = self._download(key)
        if state is None:
            logger.info("No checkpoint found at s3://%s/%s — starting fresh.", self.bucket, key)
            return None, []

        logger.info("Loaded checkpoint from s3://%s/%s", self.bucket, key)

        # ---- Model ----
        model.load(state["model_state_dict"])
        if hasattr(model, "load_optimizer_state") and "optimizer_state_dict" in state:
            model.load_optimizer_state(state["optimizer_state_dict"])

        # ---- Competence tracker ----
        t = state.get("tracker", {})
        tracker.h_ema = t.get("h_ema", tracker.h_ema)
        tracker.l_ema = t.get("l_ema", tracker.l_ema)
        tracker.step_count = t.get("step_count", tracker.step_count)

        # ---- SPL sampler ----
        spl = state.get("spl_sampler", {})
        spl_sampler._sample_losses = spl.get("sample_losses", {})

        # ---- Eval history ----
        eval_service.history = state.get("eval_history", [])

        # ---- Manager's own tracked state ----
        self.best_accuracy = state.get("best_accuracy", 0.0)

        step = state.get("step", 0)
        history = state.get("training_history", [])
        
        logger.info(
            "Resumed at step %d | best_accuracy=%.4f | C=%.3f",
            step, self.best_accuracy, tracker.competence(),
        )
        return step, history
