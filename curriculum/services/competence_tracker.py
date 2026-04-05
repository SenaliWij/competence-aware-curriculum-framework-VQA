"""
Tracks the model's evolving competence using exponential moving averages of prediction entropy (H) and training loss (L).

Computes a single competence score C hat captures how well the model currently "understands" the data.
"""

import math

class CompetenceTracker:
    """
    Maintains EMA-smoothed entropy and loss signals and derives
    a scalar competence score that drives curriculum tier selection.
    """

    def __init__(
        self,
        num_classes: int, # num of answer classes => used to compute H_max = log(num_classes)
        beta: float = 0.85, # EMA smoothing factor
        initial_loss: float = None,
        entropy_weight: float = 0.7, #alpha value
        loss_weight: float = 0.3,
        c_min: float = 0.05, # lower clamp for competence
        c_max: float = 0.99, # upper clamp for competence
        bias_correction_steps: int = 50,
    ):
        self.num_classes = num_classes
        self.beta = beta
        self.entropy_weight = entropy_weight
        self.loss_weight = loss_weight
        self.c_min = c_min
        self.c_max = c_max
        self.bias_correction_steps = bias_correction_steps

        # H_max = log(num_classes)- maximum possible entropy
        self.h_max = math.log(num_classes)

        # L0: Baseline loss; default is log(num_classes)
        self.l0 = initial_loss if initial_loss is not None else self.h_max

        # Initialised to 0.0 to support Adam-style warmup
        self.h_ema = 0.0
        self.l_ema = 0.0

        # Track how many updates we have received
        self.step_count = 0

    # Core update
    def update(self, h_batch: float, l_batch: float) -> None:
        """
        Update the running EMA estimates with a new batch's entropy and loss.

        Parameters
            h_batch : float
                Entropy of the model's softmax predictions for the current batch.
            l_batch : float
                Training loss for the current batch.

        Formula: New_EMA = (Beta * Old_EMA) + (1 - Beta) * Batch_Value
        """
        self.h_ema = self.beta * self.h_ema + (1 - self.beta) * h_batch
        self.l_ema = self.beta * self.l_ema + (1 - self.beta) * l_batch
        self.step_count += 1

    # Competence score
    def competence(self) -> float:
        """
        Compute the current competence score.
        
        C = Alpha * (Confidence) + (1 - Alpha) * (Accuracy)

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

        # Normalised entropy component (0 = max entropy -> 1 = zero entropy)
        entropy_term = 1.0 - (h_ema / self.h_max) if self.h_max > 0 else 0.0

        # Normalised loss component (0 = baseline loss -> 1 = zero loss)
        loss_term = 1.0 - (l_ema / self.l0) if self.l0 > 0 else 0.0

        # Combine both signals into one score
        c = self.entropy_weight * entropy_term + self.loss_weight * loss_term

        # Ensure score stays within [c_min, c_max] boundaries
        c = max(self.c_min, min(self.c_max, c))
        return c


