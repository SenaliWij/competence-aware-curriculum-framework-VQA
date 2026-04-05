# vilt_adapter.py
"""
Concrete ModelAdapter for HuggingFace ViLT (dandelin/vilt-b32-finetuned-vqa).
"""

import torch
import torch.nn.functional as F
from transformers import ViltForQuestionAnswering
from typing import Dict, Any

from models.base_model import ModelAdapter

# Keys expected by the HuggingFace ViLT model forward pass 
_VILT_INPUT_KEYS = (
    "input_ids",
    "attention_mask",
    "token_type_ids",
    "pixel_values",
    "pixel_mask",
    "labels",
)

class ViLTAdapter(ModelAdapter):
    """
    Wraps HuggingFace ViLT to the ModelAdapter interface.
    """

    def __init__(
        self,
        model_name: str = "dandelin/vilt-b32-finetuned-vqa",
        num_labels: int = 28,
        learning_rate: float = 5e-5,
        device: str = "cuda",
        freeze_backbone: bool = False,
    ):
        # Setup computation device
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Initialize the VQA model
        self.model = ViltForQuestionAnswering.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        ).to(self.device)

        # Freeze the backbone layers to only train the classification head
        if freeze_backbone:
            for param in self.model.vilt.parameters():
                param.requires_grad = False

        # Initialize the optimizer to only pass parameters that require gradients to the optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate,
        )

    # Helpers functions
    def _to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Filters the batch for ViLT-specific keys & moves tensors to the active device."""
        return {
            k: v.to(self.device)
            for k, v in batch.items()
            if k in _VILT_INPUT_KEYS
        }

    def _forward(self, inputs: Dict[str, Any]):
        """
        Executes the model forward pass & computes CrossEntropy loss.
        Returns: (logits, labels, loss)
        """
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs.get("token_type_ids"),
            pixel_values=inputs["pixel_values"],
            pixel_mask=inputs.get("pixel_mask"),
        )
        logits = outputs.logits
        labels = inputs["labels"]
        loss = F.cross_entropy(logits, labels)
        return logits, labels, loss

    # Abstract methods implemented from ModelAdapter
    def forward_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass only with  no backward, no optimiser step."""
        self.model.eval()
        with torch.no_grad():
            inputs = self._to_device(batch)
            logits, labels, loss = self._forward(inputs)
        return {
            "logits": logits.cpu(),
            "labels": labels.cpu(),
            "loss": loss.item(),
        }

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """        
        Execute a full  training step:
          1. Forward pass to compute logits & loss.
          2. Backward pass to compute gradients.
          3. Optimiser step to update weights.

        Parameters: Raw DataLoader batch.

        Returns:
            Dict with loss
        """
        self.model.train()
        inputs = self._to_device(batch)
        logits, labels, loss = self._forward(inputs)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def validation_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Validation step same as forward_step """
        return self.forward_step(batch)

    # Optimiser control
    def reset_optimizer(self, lr: float):
        """Re-initialise the optimiser with a new learning rate."""
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
        )

    # Persistence / Checkpointing
    def save(self, path: str):
        torch.save(self.model.state_dict(), path)

    def load(self, state_dict: Dict):
        self.model.load_state_dict(state_dict)

    def get_state_dict(self) -> Dict:
        return self.model.state_dict()

    def get_optimizer_state_dict(self) -> Dict:
        return self.optimizer.state_dict()

    def load_optimizer_state(self, state_dict: Dict):
        self.optimizer.load_state_dict(state_dict)