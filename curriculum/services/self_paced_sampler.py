# self_paced_sampler.py
"""
Soft Self-Paced Learning (SPL) sampler.

Once a tier has been selected, this module decides "which samples" within that tier to train on.  
Easier samples (lower recent loss) get higher sampling weight.
harder samples still have a non-zero chance - hence "soft" rather than hard thresholding.
"""

import math
import random
from typing import Dict, List, Optional

import torch


class SoftSelfPacedSampler:
    """
    Within-tier sampler that weights samples by recent per-sample loss.

    Parameters
        lambda_init : float
            Initial temperature (when competence is low).
        lambda_max : float
            Maximum temperature (when competence is high).
        max_loss_age : int
            Losses older than this many steps are treated as stale & ignored.
    """

    def __init__(
        self,
        lambda_init: float = 0.5,
        lambda_max: float = 5.0,
        max_loss_age: int = 500,
    ):
        self.lambda_init = lambda_init
        self.lambda_max = lambda_max
        self.max_loss_age = max_loss_age

        # dict: tier_id -> {sample_idx: (loss_value, step_recorded)}
        # Populated incrementally via record_losses() never pre-allocated.
        self._sample_losses: Dict[int, Dict[int, tuple]] = {}

    # Loss tracking
    def record_losses(
        self, tier: int, indices: List[int], losses: List[float],
        current_step: int = 0,
    ) -> None:
        """
        Store / update per-sample losses for the given tier.

        Parameters:
            tier : int
                Tier id the samples belong to.
            indices : list[int]
                Dataset indices of the samples.
            losses : list[float]
                Corresponding per-sample losses.
            current_step : int
                Current training step (used to expire stale entries).
        """
        # Initialise the tier's loss store on first encounter.
        if tier not in self._sample_losses:
            self._sample_losses[tier] = {}

        # Update per-sample loss entries.
        for idx, loss in zip(indices, losses):
            self._sample_losses[tier][idx] = (loss, current_step)

    # Temperature schedule
    def _temperature(self, competence: float) -> float:
        """
        Linear interpolation of lamda with competence.

        lamda(C) = lambda_init + C * (lambda_max - lambda_init)
        
        At C = 0.0  ->  lamda = lambda_init  (easiest samples favoured).
        At C = 1.0  ->  lamda = lambda_max   (near-uniform sampling).

        Parameters:
            competence : float
                Current competence score in [0, 1].

        Returns:
            float: Temperature lamda for use in the exponential weight formula.
        """
        return self.lambda_init + competence * (self.lambda_max - self.lambda_init)

    # Sampling
    def sample_indices(
        self,
        tier: int,
        tier_size: int,
        competence: float,
        batch_size: int,
        current_step: int = 0,
    ) -> List[int]:
        """
        Select batch_size sample indices from tier.

        Falls back to uniform random if no fresh loss history exists.
        Unknown/stale samples get the mean fresh loss so they appear neither    easy nor hard.

        Parameters:
            tier : int
                Which tier to sample from.
            tier_size : int
                Total number of samples available in this tier.
            competence : float
                Current competence score (used to set temperature).
            batch_size : int
                How many samples to draw.
            current_step : int
                Current training step (used to expire stale entries).

        Returns:
            list[int]: Selected dataset indices.
        """
        all_indices = list(range(tier_size))

        # Cannot request more samples than the tier contains.
        batch_size = min(batch_size, tier_size)

        stored = self._sample_losses.get(tier, {})
        if not stored:
            # No loss history yet -> uniform sampling
            return random.sample(all_indices, batch_size)

        # Cannot request more samples than the tier contain so only collects fresh losses
        fresh_losses = {
            idx: loss
            for idx, (loss, step_rec) in stored.items()
            if (current_step - step_rec) <= self.max_loss_age
        }
        # All recorded losses have expired fall back to uniform sampling.
        if not fresh_losses:
            return random.sample(all_indices, batch_size)

        # Assign unseen / stale samples the mean fresh loss as a neutral
        default_loss = sum(fresh_losses.values()) / len(fresh_losses)

        lam = self._temperature(competence)

        # Build weights using w_i = exp(-loss_i / lamda)
        weights = []
        for idx in all_indices:
            loss = fresh_losses.get(idx, default_loss)
            weights.append(math.exp(-loss / lam) if lam > 0 else 1.0)

        # if all weights collapsed to zero, revert to uniform sampling to avoid a zero-division in random.choices
        total_w = sum(weights)
        if total_w == 0:
            return random.sample(all_indices, batch_size)

        # Weighted sampling without replacement
        selected = set()
        pool = list(zip(all_indices, weights))
        while len(selected) < batch_size:
            # random.choices allows replacement -> loop to avoid duplicates
            picks = random.choices(
                [p[0] for p in pool],
                weights=[p[1] for p in pool],
                k=batch_size - len(selected),
            )
            selected.update(picks)

        return list(selected)[:batch_size]

    def reset_tier(self, tier: int) -> None:
        """Clear loss history for a specific tier."""
        self._sample_losses.pop(tier, None)

