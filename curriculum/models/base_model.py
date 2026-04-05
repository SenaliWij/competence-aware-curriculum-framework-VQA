# base_model.py
"""
Abstract base class (adapter pattern) for any VQA model.

The curriculum training stratergy never touches model internals directly
"""
from abc import ABC, abstractmethod
from typing import Dict, Any


class ModelAdapter(ABC):
    """
    Abstract base class for VQA models.
    """

    # Forward pass 
    @abstractmethod
    def forward_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a forward pass without updating model weights.
        
        Parameters: batch (Dict) containing input tensors.
        Returns :
            {
                "logits": Tensor,
                "loss":   float,
            }
        """
        pass

    # Training step (forward + backpropagation + optimiser step)
    @abstractmethod
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        One full training iteration (forward -> loss -> backward -> step).
        
        Parameters: batch (Dict) containing training data.
        Returns: Dict with 'logits', 'labels', and 'loss'.
        """
        pass

    # Validation step
    @abstractmethod
    def validation_step(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward pass with no gradient computation.

        Parameters: batch (Dict) containing inputs and ground-truth labels.
        Returns: Dict with 'logits', 'labels', and 'loss'.
        """
        pass

    # Persistence Methods 
    @abstractmethod
    def save(self, path: str):
        """
        Saves the current model weights to a specified file path.
        
        Parameters: path for the saved weights.
        Returns: None.
        """
        pass

    @abstractmethod
    def load(self, state_dict: Dict):
        """
        Loads model weights from a provided state dictionary.
        
        Parameters: state_dict (Dict) containing weight tensors.
        Returns: None.
        """
        pass

    @abstractmethod
    def load_optimizer_state(self, state_dict: Dict):
        """
        Restores the optimizer configuration and state.
        
        Parameters: state_dict (Dict) containing optimizer parameters.
        Returns: None.
        """
        pass

    @abstractmethod
    def get_state_dict(self) -> Dict:
        """
        Extracts the current model weights as a dictionary.

        Parameters: None.
        Returns: Dict representing the model's state.
        """
        pass

    @abstractmethod
    def get_optimizer_state_dict(self) -> Dict:
        """
        Extracts the current optimizer state as a dictionary.
        
        Parameters: None.
        Returns: Dict representing the optimizer's state.
        """
        pass
