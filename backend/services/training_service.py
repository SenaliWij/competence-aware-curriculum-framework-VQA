import time
import random
import threading
from models.schemas import TrainingMetrics, TrainingStatus, ModelConfig


class TrainingService:
    def __init__(self):
        self._is_training = False
        self._stop_event = threading.Event()
        self._global_step = 0

        self._tiers = [
            "Tier 1: Attribute & existence, no relate",
            "Tier 2: Compare attribute (multi-set), no relate",
            "Tier 3: Counting / integer compare, no relate",
            "Tier 4: Attribute & existence with relate",
            "Tier 5: Count / Compare (integer or attribute) with relate (strong composition)"
        ]

        self._tier_accuracy = {
            "tier1": 82.9,
            "tier2": 97.4,
            "tier3": 94.91,
            "tier4": 88.41,
            "tier5": 84.27
        }

        self._current_metrics = TrainingMetrics(
            global_step=10,
            loss=2.5,
            accuracy=10.0,
            current_tier=self._tiers[0],
            tier_progress=0.0,
            tier_accuracy=self._tier_accuracy
        )

        self._thread = None

    def start_training(self, config: ModelConfig):
        if self._is_training:
            return {"message": "Training already in progress"}

        self._is_training = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._training_loop, daemon=True)
        self._thread.start()
        return {"message": "Training started"}

    def stop_training(self):
        if not self._is_training:
            return {"message": "No training in progress"}

        self._stop_event.set()
        self._is_training = False
        if self._thread:
            self._thread.join()

        return {"message": "Training stopped"}

    def get_status(self) -> TrainingStatus:
        return TrainingStatus(
            is_training=self._is_training,
            status_message=(
                f"Training on {self._current_metrics.current_tier}"
                if self._is_training else "Idle"
            ),
            metrics=self._current_metrics
        )

    def _training_loop(self):
        loss = 2.5
        accuracy = 10.0
        tier_index = 0
        tier_step = 0
        steps_per_tier = 300

        while not self._stop_event.is_set():

            time.sleep(random.uniform(0.3, 1.2))

            self._global_step += 1
            tier_step += 1

            # Update global metrics
            loss = max(0.1, loss * random.uniform(0.97, 0.995))
            accuracy = min(95.0, accuracy + random.uniform(0.1, 0.6))

            # 🔥 Update current tier accuracy only
            tier_key = f"tier{tier_index + 1}"
            self._tier_accuracy[tier_key] = min(
                95.0,
                self._tier_accuracy[tier_key] + random.uniform(0.2, 0.8)
            )

            # Tier progression
            tier_progress = tier_step / steps_per_tier

            if tier_progress >= 1.0 and tier_index < len(self._tiers) - 1:
                tier_index += 1
                tier_step = 0
                tier_progress = 0.0

            self._current_metrics = TrainingMetrics(
                global_step=self._global_step,
                loss=round(loss, 4),
                accuracy=round(accuracy, 2),
                current_tier=self._tiers[tier_index],
                tier_progress=round(tier_progress, 3),
                tier_accuracy={
                    k: round(v, 2) for k, v in self._tier_accuracy.items()
                }
            )

        self._is_training = False



training_service = TrainingService()
