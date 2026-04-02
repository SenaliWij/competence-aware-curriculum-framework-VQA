# entropy.py
"""
Utility for computing the mean prediction entropy of a batch of logits.

Entropy measures how "uncertain" the model is.
    H = −Σ p_i * log(p_i)

High entropy  → model outputs are close to uniform (confused).
Low entropy   → model outputs are peaked (confident).
"""

import torch


def batch_entropy(logits: torch.Tensor) -> float:
    """
    Compute the mean entropy over a batch of logit vectors.

    Parameters
    ----------
    logits : Tensor of shape [batch_size, num_classes]
        Raw (pre-softmax) model outputs.

    Returns
    -------
    float
        Mean entropy across the batch (in nats, i.e. natural-log base).
    """
    # Softmax → probability distribution per sample
    probs = torch.softmax(logits, dim=-1)

    # Clamp to avoid log(0)
    probs = probs.clamp(min=1e-9)

    # Per-sample entropy: H_i = −Σ_c p_ic * log(p_ic)
    per_sample = -(probs * probs.log()).sum(dim=-1)   # shape: [batch_size]

    return per_sample.mean().item()
