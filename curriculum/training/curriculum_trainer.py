# curriculum_trainer.py
"""
Main training loop 
For each batch:
  1. Compute competence C from EMA entropy & loss.
  2. Use power-law scoring to build a probability distribution over tiers.
  3. Sample a tier.
  4. Use soft self-paced learning to pick samples within the tier.
  5. Train on the selected samples.
  6. Periodically validate and checkpoint .

"""

import time
import logging
from collections import defaultdict
from typing import Dict, List, Optional, Any

import torch
from torch.utils.data import DataLoader, Subset

from models.base_model import ModelAdapter
from services.competence_tracker import CompetenceTracker
from services.tier_sampler import TierSampler
from services.self_paced_sampler import SoftSelfPacedSampler
from services.entropy import batch_entropy
from services.evaluation_service import EvaluationService
from services.checkpoint_service import CheckpointManager

logger = logging.getLogger(__name__)


class CurriculumTrainer:
    """
    Runs the competence-aware curriculum training loop.

    Parameters
        model : Any model implementing the ModelAdapter interface.
        tier_datasets : Maps tier_id -> PyTorch Dataset for that tier.
        val_dataset : Validation dataset (single or multi-tier).
        num_classes : Number of answer classes (used by CompetenceTracker).
        checkpoint_manager: Centralised S3 checkpoint service.
        batch_size : Training batch size.
        num_steps : Total number of training steps (batches) to run.
        beta : EMA smoothing factor for the competence tracker.
        entropy_weight : Weight given to entropy term in competence score.
        loss_weight : Weight given to loss term in competence score.
        collate_fn : Custom DataLoader collate function (optional).
        difficulty : Tier difficulty map passed to TierSampler (optional).
        val_every : Run validation every N steps.
        checkpoint_every : Save a checkpoint every N steps.
        log_every : Print a log line every N steps.
        tier_val_datasets: Per-tier validation datasets for granular eval.
    """

    def __init__(
        self,
        model: ModelAdapter,
        tier_datasets: Dict[int, Any],
        val_dataset: Any,
        num_classes: int,
        checkpoint_manager: CheckpointManager,
        batch_size: int = 32,
        num_steps: int = 10000,
        beta: float = 0.85,
        entropy_weight: float = 0.7,
        loss_weight: float = 0.3,
        collate_fn=None,
        difficulty: Optional[Dict[int, float]] = None,
        val_every: int = 1000,
        checkpoint_every: int = 200,
        log_every: int = 50,
        tier_val_datasets: Optional[Dict[int, Any]] = None,
    ):
        self.model = model
        self.tier_datasets = tier_datasets
        self.val_dataset = val_dataset
        self.tier_val_datasets = tier_val_datasets or {}
        self.batch_size = batch_size
        self.num_steps = num_steps
        self.collate_fn = collate_fn
        self.val_every = val_every
        self.checkpoint_every = checkpoint_every
        self.log_every = log_every

        # Checkpoint Manager
        self.checkpoint = checkpoint_manager

        # Curriculum components
         # Tracks EMA entropy and loss to derive a scalar competence score
        self.tracker = CompetenceTracker(
            num_classes=num_classes,
            beta=beta,
            entropy_weight=entropy_weight,
            loss_weight=loss_weight,
        )
         # Converts competence -> tier probability distribution.
        self.tier_sampler = TierSampler(difficulty=difficulty)
        # Weights within-tier samples by recent per-sample loss
        self.spl_sampler = SoftSelfPacedSampler()
        # Records validation metric history.
        self.eval_service = EvaluationService()

        # Pre-compute tier sizes for the SPL sampler
        self.tier_sizes: Dict[int, int] = {
            t: len(ds) for t, ds in tier_datasets.items()
        }

        # Full per-step log, returned at the end of train() and checkpointed
        self.history: List[Dict[str, Any]] = []

        # Accumulators reset every log_every steps for interval summaries
        self._tier_selection_counts: Dict[int, int] = defaultdict(int)
        self._interval_losses: List[float] = []
        self._interval_entropies: List[float] = []

    # Resume from checkpoint
    def resume(self, tag: str = "latest") -> int:
        """
        Attempt to load a checkpoint from S3 and restore all component
        states via the CheckpointManager.

        Returns
        -------
        int
            The step to resume from (0 if no checkpoint found).
        """
        result = self.checkpoint.load(
            model=self.model,
            tracker=self.tracker,
            spl_sampler=self.spl_sampler,
            eval_service=self.eval_service,
            tag=tag,
        )
        if result[0] is not None:
            step, history = result
            self.history = history # Restore the training log
            return step
        return 0

    # Log training config at startup
    def _log_config(self):
        """Print all key training parameters at startup."""
        logger.info("=" * 65)
        logger.info("  CURRICULUM TRAINING CONFIGURATION")
        logger.info("=" * 65)
        logger.info("  Total steps        : %d", self.num_steps)
        logger.info("  Batch size         : %d", self.batch_size)
        logger.info("  Validate every     : %d steps", self.val_every)
        logger.info("  Checkpoint every   : %d steps", self.checkpoint_every)
        logger.info("  Log every          : %d steps", self.log_every)
        logger.info("-" * 65)
        logger.info("  EMA beta           : %.3f", self.tracker.beta)
        logger.info("  H_max              : %.4f", self.tracker.h_max)
        logger.info("  L0 (baseline loss) : %.4f", self.tracker.l0)
        logger.info("  Entropy weight (α) : %.2f", self.tracker.entropy_weight)
        logger.info("  Loss weight (1-α)  : %.2f", self.tracker.loss_weight)
        logger.info("  C clamp range      : [%.2f, %.2f]", self.tracker.c_min, self.tracker.c_max)
        logger.info("-" * 65)
        logger.info("  SPL λ_init         : %.2f", self.spl_sampler.lambda_init)
        logger.info("  SPL λ_max          : %.2f", self.spl_sampler.lambda_max)
        logger.info("-" * 65)
        diff = self.tier_sampler.difficulty
        for t in sorted(diff):
            logger.info("  Tier %d  |  difficulty=%.1f  |  samples=%d",
                        t, diff[t], self.tier_sizes.get(t, 0))
        logger.info("=" * 65)

    # Main training loop
    def train(self, start_step: int = 0) -> List[Dict[str, Any]]:
        """
        Execute the curriculum training loop from start to end step.

        Each step:
          1.  Compute current competence C.
          2.  Sample a tier using competence-driven power-law probabilities.
          3.  Select a batch within the tier using soft-SPL weights.
          4.  Run a forward pass to measure entropy and loss.
          5.  Update the EMA competence tracker.
          6.  Record per-sample losses for future SPL weighting.
          7.  Run the actual training step (backprop + optimiser).
          8.  Log, validate, and checkpoint on their respective schedules.

        Parameters:
            start_step: Step to begin from.
        Returns:
            Full per-step history list .
        """
        self._log_config()
        logger.info("Starting training from step %d -> %d", start_step, self.num_steps)

        training_start_time = time.time()
        interval_start_time = time.time()

        for step in range(start_step, self.num_steps):

            step_start = time.time()

            # 1) Current competence
            competence = self.tracker.competence()

            # 2) Tier probabilities & sample a tier
            tier_probs = self.tier_sampler.tier_probabilities(competence)
            tier = self.tier_sampler.sample_tier(competence)

            # Track tier selection frequency
            self._tier_selection_counts[tier] += 1

            # 3) Within-tier Soft Self Paced Learning sampling
            spl_lambda = self.spl_sampler._temperature(competence)
            indices = self.spl_sampler.sample_indices(
                tier=tier,
                tier_size=self.tier_sizes[tier],
                competence=competence,
                batch_size=self.batch_size,
                current_step=step,
            )

            # 4) Build a mini-batch from the selected indices
            subset = Subset(self.tier_datasets[tier], indices)
            loader = DataLoader(
                subset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
            )
            batch = next(iter(loader))

            # 5) Forward pass to get logits for entropy & loss
            fwd = self.model.forward_step(batch)
            h_batch = batch_entropy(fwd["logits"])
            l_batch = fwd["loss"]

            # 6) Update EMA tracker with fresh entropy and loss
            self.tracker.update(h_batch, l_batch)

            # 7) Record per-sample losses for next SPL sampling round
            self._record_per_sample_losses(batch, tier, indices, step=step)

            # 8) Actual training step -> forward + backward + optimiser update
            train_out = self.model.train_step(batch)

            step_time = time.time() - step_start

            # Accumulate interval metrics for the  summary log.
            self._interval_losses.append(train_out["loss"])
            self._interval_entropies.append(h_batch)

            # 9) Record this step in the training history
            entry = {
                "step": step,
                "tier": tier,
                "competence": competence,
                "h_batch": h_batch,
                "h_ema": self.tracker.h_ema,
                "l_batch": l_batch,
                "l_ema": self.tracker.l_ema,
                "train_loss": train_out["loss"],
                "tier_probs": tier_probs,
                "spl_lambda": spl_lambda,
                "step_time_s": step_time,
            }
            self.history.append(entry)

            # 10) Per-step log
            if step % self.log_every == 0:
                prob_str = " ".join(
                    f"T{t}:{p:.3f}" for t, p in sorted(tier_probs.items())
                )
                logger.info(
                    "[Step %5d/%d]  C=%.4f | tier=%d | "
                    "train_loss=%.4f | H_batch=%.4f  H_ema=%.4f | "
                    "L_batch=%.4f  L_ema=%.4f | SPL_λ=%.2f | %.2fs",
                    step, self.num_steps, competence, tier,
                    train_out["loss"], h_batch, self.tracker.h_ema,
                    l_batch, self.tracker.l_ema, spl_lambda, step_time,
                )
                logger.info(
                    "           Tier probs: %s", prob_str,
                )

                # Periodic summary every log_every steps
                if self._interval_losses:
                    avg_loss = sum(self._interval_losses) / len(self._interval_losses)
                    avg_ent = sum(self._interval_entropies) / len(self._interval_entropies)
                    elapsed = time.time() - interval_start_time
                    steps_per_sec = len(self._interval_losses) / max(elapsed, 1e-6)

                    total_selections = sum(self._tier_selection_counts.values())
                    tier_pct_str = " ".join(
                        f"T{t}:{self._tier_selection_counts[t]/max(total_selections,1)*100:.1f}%"
                        for t in sorted(self._tier_selection_counts)
                    )
                    logger.info(
                        "           Interval avg: loss=%.4f  entropy=%.4f | "
                        "%.2f steps/s | Tier usage: %s",
                        avg_loss, avg_ent, steps_per_sec, tier_pct_str,
                    )

                    # Reset interval trackers
                    self._interval_losses.clear()
                    self._interval_entropies.clear()
                    interval_start_time = time.time()

            # 11) Periodic validation
            val_accuracy = 0.0
            if step > 0 and step % self.val_every == 0:
                val_accuracy = self._validate(step)

            # 12) Periodic checkpoint to S3
            if step > 0 and step % self.checkpoint_every == 0:
                self.checkpoint.save(
                    step=step,
                    model=self.model,
                    tracker=self.tracker,
                    spl_sampler=self.spl_sampler,
                    eval_service=self.eval_service,
                    accuracy=val_accuracy,
                    history=self.history,
                )

        # Final validation + checkpoint
        final_acc = self._validate(self.num_steps)
        self.checkpoint.save(
            step=self.num_steps,
            model=self.model,
            tracker=self.tracker,
            spl_sampler=self.spl_sampler,
            eval_service=self.eval_service,
            accuracy=final_acc,
            history=self.history,
        )

        total_time = time.time() - training_start_time
        self._log_final_summary(total_time, final_acc)
        return self.history

    # End of training summary
    def _log_final_summary(self, total_time: float, final_acc: float):
        """Print a comprehensive summary at the end of training."""
        logger.info("=" * 65)
        logger.info("  TRAINING COMPLETE")
        logger.info("=" * 65)
        logger.info("  Total steps      : %d", len(self.history))
        logger.info("  Total time       : %.1f s  (%.2f min)",
                     total_time, total_time / 60)
        logger.info("  Avg step time    : %.3f s",
                     total_time / max(len(self.history), 1))
        logger.info("-" * 65)
        logger.info("  Final competence : %.4f", self.tracker.competence())
        logger.info("  Final H_ema      : %.4f  (H_max=%.4f)",
                     self.tracker.h_ema, self.tracker.h_max)
        logger.info("  Final L_ema      : %.4f  (L0=%.4f)",
                     self.tracker.l_ema, self.tracker.l0)
        logger.info("  Final val acc    : %.4f", final_acc)
        logger.info("  Best val acc     : %.4f", self.checkpoint.best_accuracy)
        logger.info("-" * 65)

        # Tier usage breakdown
        total_selections = sum(self._tier_selection_counts.values()) or 1
        logger.info("  Tier selection breakdown (total %d batches):", total_selections)
        for t in sorted(self._tier_selection_counts):
            count = self._tier_selection_counts[t]
            pct = count / total_selections * 100
            logger.info("    Tier %d : %5d batches  (%5.1f%%)", t, count, pct)

        # Validation history summary
        if self.eval_service.history:
            accs = [h["accuracy"] for h in self.eval_service.history]
            losses = [h["loss"] for h in self.eval_service.history]
            logger.info("-" * 65)
            logger.info("  Validation summary (%d evals):", len(accs))
            logger.info("    Accuracy - min=%.4f  max=%.4f  last=%.4f",
                         min(accs), max(accs), accs[-1])
            logger.info("    Loss     - min=%.4f  max=%.4f  last=%.4f",
                         min(losses), max(losses), losses[-1])

        logger.info("=" * 65)


    # Validation
    def _validate(self, step: int) -> float:
        """
        Run a full validation pass and log results, including per-tier accuracy.

        Returns:
            Validation accuracy.
        """
        logger.info("-" * 45)
        logger.info("[Validation @ step %d]  Running...", step)
        val_start = time.time()

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        num_batches = 0

        for batch in val_loader:
            out = self.model.validation_step(batch)
            logits = out["logits"]
            labels = out["labels"]

            # Weight loss by sample count for a true dataset-level average.
            total_loss += out["loss"] * logits.size(0)
            preds = logits.argmax(dim=-1)
            if labels.dim() == 2:
                # Soft / one-hot labels -> convert to hard class index.
                labels = labels.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_samples += logits.size(0)
            num_batches += 1

        avg_loss = total_loss / max(total_samples, 1)
        accuracy = total_correct / max(total_samples, 1)
        val_time = time.time() - val_start
        self.eval_service.record_metrics(avg_loss, accuracy, step=step)

        # Compute improvement delta from previous validation
        prev_acc = 0.0
        prev_loss = 0.0
        if len(self.eval_service.history) >= 2:
            prev_acc = self.eval_service.history[-2]["accuracy"]
            prev_loss = self.eval_service.history[-2]["loss"]
        acc_delta = accuracy - prev_acc
        loss_delta = avg_loss - prev_loss

        is_best = accuracy > self.checkpoint.best_accuracy
        best_marker = "  NEW BEST" if is_best else ""

        logger.info(
            "[Validation @ step %d]  "
            "loss=%.4f (Δ%+.4f) | accuracy=%.4f (Δ%+.4f) | "
            "C=%.4f | %d samples, %d batches | %.1fs%s",
            step, avg_loss, loss_delta, accuracy, acc_delta,
            self.tracker.competence(), total_samples,
            num_batches, val_time, best_marker,
        )

        # Per-tier validation
        self._validate_per_tier(step)

        logger.info("-" * 45)

        return accuracy

    # Per-tier validation
    def _validate_per_tier(self, step: int):
        """
        Run validation separately for each tier and log per-tier accuracy.
        """
        if not self.tier_val_datasets:
            return

        tier_accs = {}
        tier_losses = {}

        for tier_id in sorted(self.tier_val_datasets.keys()):
            ds = self.tier_val_datasets[tier_id]
            loader = DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=self.collate_fn,
            )

            t_correct = 0
            t_total = 0
            t_loss = 0.0

            for batch in loader:
                out = self.model.validation_step(batch)
                logits = out["logits"]
                labels = out["labels"]

                preds = logits.argmax(dim=-1)
                if labels.dim() == 2:
                    labels = labels.argmax(dim=-1)

                t_correct += (preds == labels).sum().item()
                t_total += logits.size(0)
                t_loss += out["loss"] * logits.size(0)

            acc = t_correct / max(t_total, 1)
            loss = t_loss / max(t_total, 1)
            tier_accs[tier_id] = acc
            tier_losses[tier_id] = loss

        # Log a single compact line with per-tier accuracies
        acc_str = "  ".join(
            f"T{t}_acc={tier_accs[t]:.4f}" for t in sorted(tier_accs)
        )
        loss_str = "  ".join(
            f"T{t}_loss={tier_losses[t]:.4f}" for t in sorted(tier_losses)
        )
        logger.info("[Per-tier @ step %d]  %s", step, acc_str)
        logger.info("[Per-tier @ step %d]  %s", step, loss_str)

    # Per-sample loss tracking (for SPL)
    def _record_per_sample_losses(
        self, batch: Dict[str, Any], tier: int, indices: List[int],
        step: int = 0,
    ):
        """Compute per-sample CE loss and feed it to the SPL sampler."""
        logits = self.model.forward_step(batch)["logits"]
        labels = batch["labels"]

        if labels.dim() == 2:
            labels = labels.argmax(dim=-1)

        # Per-sample cross-entropy (no reduction)
        per_sample_loss = torch.nn.functional.cross_entropy(
            logits, labels, reduction="none"
        )

        self.spl_sampler.record_losses(
            tier=tier,
            indices=indices,
            losses=per_sample_loss.tolist(),
            current_step=step,
        )
