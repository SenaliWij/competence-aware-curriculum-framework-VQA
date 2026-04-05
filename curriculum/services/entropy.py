# entropy.py
"""
Utility for computing the mean prediction entropy of a batch of logits.

Entropy measures how "uncertain" the model is.

High entropy  -> model outputs are close to uniform. 
Low entropy   -> model outputs are peaked.
"""

import torch

def batch_entropy(logits: torch.Tensor) -> float:
    """
    Compute the mean entropy over a batch of logit vectors.

    Parameters:
        logits : Tensor of shape [batch_size, num_classes]

    Returns:
        float: Mean entropy across the batch .
    """
    # Convert raw scores (logits) into a probability distribution that sums to 1
    probs = torch.softmax(logits, dim=-1)

    # Clamp to avoid log(0)
    probs = probs.clamp(min=1e-9)

    # Calculate Shannon Entropy per sample
    per_sample = -(probs * probs.log()).sum(dim=-1)   # shape: [batch_size]

    # Average the entropy across the entire batch to return a single scalar value
    return per_sample.mean().item()
