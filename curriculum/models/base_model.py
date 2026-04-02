# base_model.py
"""
Abstract base class (adapter pattern) for any VQA model.

The curriculum framework never touches model internals directly —
it only calls the methods defined here.  To plug in a new model
(BLIP-2, OFA, …), create a subclass of ModelAdapter.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any


class ModelAdapter(ABC):
    """
    Common interface that every VQA model must implement.
    """

    # ------------------------------------------------------------------
    # Forward pass (no weight update)
    # ------------------------------------------------------------------
    @abstractmethod
    def forward_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a forward pass **without** back-propagation.

        Must return at least:
            {
                "logits": Tensor [batch_size, num_classes],
                "loss":   float  (scalar CE / task loss),
            }

        This is used by the curriculum loop to compute entropy and
        per-sample losses *before* deciding whether to update weights.
        """
        pass

    # ------------------------------------------------------------------
    # Training step (forward + backward + optimiser)
    # ------------------------------------------------------------------
    @abstractmethod
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        One full training iteration (forward → loss → backward → step).

        Returns:
            {"loss": float, ...}   (may include extra metrics)
        """
        pass

    # ------------------------------------------------------------------
    # Validation step
    # ------------------------------------------------------------------
    @abstractmethod
    def validation_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass with no gradient computation.

        Returns:
            {"logits": Tensor, "labels": Tensor, "loss": float}
        """
        pass

    # ------------------------------------------------------------------
    # Test step
    # ------------------------------------------------------------------
    @abstractmethod
    def test_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Inference pass for data without labels.

        Returns:
            {"logits": Tensor, "preds": Tensor}
        """
        pass

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    @abstractmethod
    def save(self, path: str):
        """Save model weights to disk / cloud."""
        pass

    @abstractmethod
    def load(self, state_dict: Dict):
        """Load model weights from state_dict."""
        pass

    @abstractmethod
    def load_optimizer_state(self, state_dict: Dict):
        """Load optimiser state from state_dict."""
        pass

    @abstractmethod
    def get_state_dict(self) -> Dict:
        """Return model state dict (for checkpointing)."""
        pass

    @abstractmethod
    def get_optimizer_state_dict(self) -> Dict:
        """Return optimiser state dict (for checkpointing)."""
        pass
