# evaluation_service.py
import torch
import numpy as np
from typing import List, Dict

class EvaluationService:
    """
    Computes and aggregates metrics to assess model competence.
    """
    def __init__(self, history_window: int = 5):
        self.history_window = history_window
        # History is a list of dicts: [{'loss': float, 'accuracy': float}]
        self.history: List[Dict[str, float]] = []

    def record_metrics(self, loss: float, accuracy: float, step: int = None):
        """
        Records metrics from one validation step or epoch.

        Parameters:
            loss: validation loss value
            accuracy: validation accuracy value
            step: optional global training step
        """
        entry = {'loss': loss, 'accuracy': accuracy}
        if step is not None:
            entry['step'] = step
        self.history.append(entry)
