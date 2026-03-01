
import torch
from transformers import ViltForQuestionAnswering
from typing import Dict, Any
from models.base_model import ModelAdapter


class ViLTAdapter(ModelAdapter):
    """
    Adapter class that wraps the HuggingFace ViLT model so that it follows the ModelAdapter interface.

    """
    def __init__(
        self,
        model_name: str = "dandelin/vilt-b32-finetuned-vqa",
        num_labels: int = 28,
        learning_rate: float = 5e-5,
        device: str = "cuda",
        freeze_backbone: bool = False
    ):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load pretrained ViLT model for VQA
        self.model = ViltForQuestionAnswering.from_pretrained(
            model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True
        ).to(self.device)

        # Optionally freeze the backbone (ViLT encoder layers)
        if freeze_backbone:
            for param in self.model.vilt.parameters():
                param.requires_grad = False

        # Initialize AdamW optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=learning_rate
        )


    # TRAIN STEP
    def train_step(self, batch):

        """
        Performs one training iteration:
            - Forward pass
            - Loss computation
            - Backward pass
            - Optimizer update
        """
        self.model.train()

        # Move required inputs to correct device
        # Only keep relevant keys expected by ViLT
        inputs = {
            k: v.to(self.device)
            for k, v in batch.items()
            if k in (
                "input_ids",
                "attention_mask",
                "token_type_ids",
                "pixel_values",
                "pixel_mask",
                "labels"
            )
        }
    
        # Reset gradients from previous step
        self.optimizer.zero_grad()
    
        # Forward pass through the model
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            token_type_ids=inputs.get("token_type_ids"),
            pixel_values=inputs["pixel_values"],
            pixel_mask=inputs.get("pixel_mask"),
        )
    
        logits = outputs.logits
        labels = inputs["labels"]
    
        # Compute cross-entropy loss for classification
        loss = torch.nn.functional.cross_entropy(logits, labels)

        # Backpropagation
        loss.backward()

        # Update parameters
        self.optimizer.step()
    
        return {"loss": loss.item()}

    # VALIDATION STEP
    def validation_step(self, batch):
        """
        Performs validation without updating model weights.
        """
        self.model.eval()

        # Disable gradient computation for efficiency
        with torch.no_grad():
            inputs = {
                k: v.to(self.device)
                for k, v in batch.items()
                if k in (
                    "input_ids",
                    "attention_mask",
                    "token_type_ids",
                    "pixel_values",
                    "pixel_mask",
                    "labels"
                )
            }
    
            outputs = self.model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                token_type_ids=inputs.get("token_type_ids"),
                pixel_values=inputs["pixel_values"],
                pixel_mask=inputs.get("pixel_mask"),
            )
    
            logits = outputs.logits
            labels = inputs["labels"]

            # Compute validation loss
            loss = torch.nn.functional.cross_entropy(logits, labels)
    
        return {
            "logits": logits.cpu(),
            "labels": labels.cpu(),
            "loss": loss.item()
        }


    # OPTIMIZER CONTROL

    def reset_optimizer(self, lr: float):
        """
        Reinitializes the optimizer with a new learning rate.
        Useful when moving between curriculum tiers.
        """
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr
        )
    
    def save(self, path: str):
        """
        Saves only model weights.
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path: str):
        """
        Loads model weights from file.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    # CHECKPOINTING

    def get_state_dict(self):
        """
        Returns model weights (used when saving full checkpoints).
        """
        return self.model.state_dict()

    def get_optimizer_state_dict(self):
        """
        Returns optimizer state (needed to resume training).
        """
        return self.optimizer.state_dict()

    def load_optimizer_state(self, state_dict):
        """
        Loads optimizer state from checkpoint.
        """
        self.optimizer.load_state_dict(state_dict)