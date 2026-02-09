import time
import random
import threading
from models.schemas import TrainingMetrics, TrainingStatus, ModelConfig

class TrainingService:
    def __init__(self):
        self._is_training = False
        self._stop_event = threading.Event()
        self._current_metrics = TrainingMetrics(
            epoch=0, global_step=0, loss=2.5, accuracy=0.0, current_tier="Tier 1: Object Recognition", tier_progress=0.0
        )
        self._thread = None
        
    def start_training(self, config: ModelConfig):
        if self._is_training:
            return {"message": "Training already in progress"}
        
        self._is_training = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._training_loop, args=(config,))
        self._thread.start()
        return {"message": "Training started"}

    def stop_training(self):
        if self._is_training:
            self._stop_event.set()
            self._is_training = False
            if self._thread:
                self._thread.join()
            return {"message": "Training stopped"}
        return {"message": "No training in progress"}

    def get_status(self) -> TrainingStatus:
        return TrainingStatus(
            is_training=self._is_training,
            status_message="Training models on Tier " + self._current_metrics.current_tier if self._is_training else "Idle",
            metrics=self._current_metrics
        )

    def _training_loop(self, config: ModelConfig):
        tiers = [
            "Tier 1: Object Recognition",
            "Tier 2: Attribute Identification",
            "Tier 3: Relation Reasoning",
            "Tier 4: Complex Logic",
            "Tier 5: General VQA"
        ]
        
        total_epochs = config.epochs
        params = {
            "loss": 2.5,
            "accuracy": 10.0
        }
        
        for epoch in range(1, total_epochs + 1):
            if self._stop_event.is_set():
                break
                
            tier_idx = min(len(tiers) - 1, int((epoch / total_epochs) * len(tiers)))
            current_tier = tiers[tier_idx]
            
            steps_per_epoch = 100
            for step in range(steps_per_epoch):
                if self._stop_event.is_set():
                    break
                
                time.sleep(0.05) # Simulate processing time
                
                # Update mock metrics
                params["loss"] = max(0.1, params["loss"] * 0.99 + random.uniform(-0.05, 0.05))
                params["accuracy"] = min(95.0, params["accuracy"] + 0.5 + random.uniform(-0.1, 0.4))
                
                self._current_metrics = TrainingMetrics(
                    epoch=epoch,
                    global_step=(epoch - 1) * steps_per_epoch + step,
                    loss=round(params["loss"], 4),
                    accuracy=round(params["accuracy"], 2),
                    current_tier=current_tier,
                    tier_progress=step / steps_per_epoch
                )

        self._is_training = False

training_service = TrainingService()
