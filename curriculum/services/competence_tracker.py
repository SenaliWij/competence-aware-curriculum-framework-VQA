# competence_tracker.py
"""
Tracks the model's evolving competence using exponential moving averages
of prediction entropy (H) and training loss (L).

Computes a single competence score C ∈ [0.05, 0.99] that captures
how well the model currently "understands" the data.
"""

import math
from typing import Dict


class CompetenceTracker:
    """
    Maintains EMA-smoothed entropy and loss signals and derives
    a scalar competence score that drives curriculum tier selection.

    Parameters
    ----------
    num_classes : int
        Number of answer classes (used to compute H_max = log(num_classes)).
    beta : float
        EMA smoothing factor (0 < beta < 1). Higher = more smoothing.
    initial_loss : float
        L0, the baseline loss used to normalise the loss component.
        A reasonable default is log(num_classes) ≈ loss at uniform guessing.
    entropy_weight : float
        Weight (α) for the entropy component in the competence formula.
    loss_weight : float
        Weight (1 − α) for the loss component.
    c_min : float
        Lower clamp for competence.
    c_max : float
        Upper clamp for competence.
    """

    def __init__(
        self,
        num_classes: int,
        beta: float = 0.9,
        initial_loss: float = None,
        entropy_weight: float = 0.7,
        loss_weight: float = 0.3,
        c_min: float = 0.05,
        c_max: float = 0.99,
        bias_correction_steps: int = 50,
    ):
        self.num_classes = num_classes
        self.beta = beta
        self.entropy_weight = entropy_weight
        self.loss_weight = loss_weight
        self.c_min = c_min
        self.c_max = c_max
        self.bias_correction_steps = bias_correction_steps

        # H_max = log(num_classes) — maximum possible entropy (uniform dist)
        self.h_max = math.log(num_classes)

        # L0 — baseline loss; default to H_max (random-guessing CE loss)
        self.l0 = initial_loss if initial_loss is not None else self.h_max

        # Running EMA values — initialised to 0.0 to support Adam-style bias correction
        self.h_ema = 0.0
        self.l_ema = 0.0

        # Track how many updates we have received
        self.step_count = 0

    # ------------------------------------------------------------------
    # Core update
    # ------------------------------------------------------------------
    def update(self, h_batch: float, l_batch: float) -> None:
        """
        Update the running EMA estimates with a new batch's entropy and loss.

        Parameters
        ----------
        h_batch : float
            Entropy of the model's softmax predictions for the current batch.
        l_batch : float
            Training loss for the current batch.
        """
        self.h_ema = self.beta * self.h_ema + (1 - self.beta) * h_batch
        self.l_ema = self.beta * self.l_ema + (1 - self.beta) * l_batch
        self.step_count += 1


    # Competence score
   
    def competence(self) -> float:
        """
        Compute the current competence score.

        C = α * (1 − H_corrected/H_max) + (1−α) * (1 − L_corrected/L0)

        For the first `bias_correction_steps`, EMA values are bias-corrected
        using Adam-style correction:  x_corrected = x_ema / (1 − β^t)
        This prevents the initial values from inflating C prematurely.

        Clamped to [c_min, c_max].
        """
        # Before the first batch is processed, assume minimum competence.
        if self.step_count == 0:
            return self.c_min

        h_ema = self.h_ema
        l_ema = self.l_ema

        # Apply EMA bias correction for early steps (like Adam's bias correction)
        if self.step_count > 0 and self.step_count <= self.bias_correction_steps:
            correction = 1.0 - (self.beta ** self.step_count)
            h_ema = self.h_ema / correction
            l_ema = self.l_ema / correction

        # Normalised entropy component (0 = max entropy → 1 = zero entropy)
        entropy_term = 1.0 - (h_ema / self.h_max) if self.h_max > 0 else 0.0

        # Normalised loss component (0 = baseline loss → 1 = zero loss)
        loss_term = 1.0 - (l_ema / self.l0) if self.l0 > 0 else 0.0

        c = self.entropy_weight * entropy_term + self.loss_weight * loss_term

        # Clamp
        c = max(self.c_min, min(self.c_max, c))
        return c


