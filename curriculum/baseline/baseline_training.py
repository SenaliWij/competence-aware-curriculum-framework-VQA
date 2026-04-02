import time
import logging
import os
import sys
import random
import numpy as np

# Add parent directory to sys.path to allow importing from other directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from torch.utils.data import DataLoader
from transformers import ViltProcessor

from models.vilt_adapter import ViLTAdapter
from services.checkpoint_service import CheckpointManager
from baseline.baseline_utils import load_answer_vocab, CLEVRBaselineDatasetS3, vilt_collate_fn

logger = logging.getLogger(__name__)

# =====================================================================
# CONFIGURATION
# =====================================================================

CONFIG = {
    # ---- General ----
    "seed":                42,

    # ---- S3 ----
    "s3_bucket":           "clevr-curriculum",
    "s3_images_prefix":    "dataset/images",
    "s3_questions_prefix": "dataset/questions",

    # ---- Answer vocab (local JSON) ----
     "answer_vocab_path": "data/answer_vocab.json",

    # ---- Dataset ----
    "max_question_length": 32,
    "max_train_samples":   320000, 

    # ---- Model ----
    "model_name":          "dandelin/vilt-b32-finetuned-vqa",
    "learning_rate":       5e-5,
    "freeze_backbone":     False,
    "device":              "cuda",   # "cuda" or "cpu"

    # ---- Training loop ----
    "batch_size":          32,
    "num_epochs":          1,        # Baseline might be trained for epochs or steps
    "log_every":           50,       # print log every N steps
    "save_every":          5000,     # save checkpoint every N steps
    "run_name":            "baseline_training_run",
    "checkpoint_prefix":   "checkpoints/baseline",
}

def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

# =====================================================================
# MAIN ROUTINE
# =====================================================================

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    _setup_logging()
    
    cfg = CONFIG
    set_seed(cfg.get("seed", 42))
    processor = ViltProcessor.from_pretrained(cfg["model_name"])
    answer2id = load_answer_vocab(cfg)
    num_classes = len(answer2id)

    # 1) Datasets
    logger.info("Initializing baseline train dataset...")
    train_dataset = CLEVRBaselineDatasetS3(
        bucket=cfg["s3_bucket"],
        images_prefix=cfg["s3_images_prefix"],
        questions_prefix=cfg["s3_questions_prefix"],
        processor=processor,
        filename="CLEVR_baseline_train_questions.json", 
        answer2id=answer2id,
        max_length=cfg["max_question_length"],
        max_samples=cfg["max_train_samples"], 
    )

    logger.info("Initializing baseline validation dataset...")
    val_dataset = CLEVRBaselineDatasetS3(
        bucket=cfg["s3_bucket"],
        images_prefix=cfg["s3_images_prefix"],
        questions_prefix=cfg["s3_questions_prefix"],
        processor=processor,
        filename="CLEVR_baseline_val_questions.json",
        answer2id=answer2id,
        max_length=cfg["max_question_length"],
    )

    train_loader = DataLoader(
        train_dataset, 
        batch_size=cfg["batch_size"], 
        shuffle=True, 
        collate_fn=vilt_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=cfg["batch_size"], 
        shuffle=False, 
        collate_fn=vilt_collate_fn
    )

    # 2) Model
    logger.info(f"Loaded Model: {cfg['model_name']} (classes={num_classes})")
    model = ViLTAdapter(
        model_name=cfg["model_name"],
        num_labels=num_classes,
        learning_rate=cfg["learning_rate"],
        device=cfg["device"],
        freeze_backbone=cfg["freeze_backbone"],
    )

    checkpoint_manager = CheckpointManager(
        bucket=cfg["s3_bucket"],
        run_name=cfg.get("run_name", "baseline_run"),
        prefix=cfg.get("checkpoint_prefix", "checkpoints/baseline")
    )

    # 3) Training Loop
    logger.info("=" * 60)
    logger.info(f"Starting Baseline Training for {len(train_dataset)} samples")
    logger.info("=" * 60)

    train_start = time.time()
    
    # We do a basic epoch-based loop. (1 epoch since it's 320k samples)
    step = 0
    for epoch in range(cfg["num_epochs"]):
        for batch in train_loader:
            step_start = time.time()
            out = model.train_step(batch)
            loss = out["loss"]
            step_time = time.time() - step_start
            
            if step % cfg["log_every"] == 0:
                logger.info(f"[Epoch {epoch} | Step {step}] loss={loss:.4f} | {step_time:.2f}s")
            
            if step > 0 and "save_every" in cfg and step % cfg["save_every"] == 0:
                logger.info(f"Saving intermediate baseline checkpoint at step {step}...")
                checkpoint_manager.save_baseline(
                    step=step,
                    model=model,
                    save_numbered=True
                )
            
            step += 1

    total_time = time.time() - train_start
    logger.info(f"Training complete. Time taken: {total_time:.2f}s")
    
    # 4) Validation at the end
    logger.info("-" * 45)
    logger.info("[Validation END] Running...")
    val_start = time.time()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    num_batches = 0

    for batch in val_loader:
        out = model.validation_step(batch)
        logits = out["logits"]
        labels = out["labels"]

        total_loss += out["loss"] * logits.size(0)
        preds = logits.argmax(dim=-1)
        if labels.dim() == 2:
            labels = labels.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_samples += logits.size(0)
        num_batches += 1

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    val_time = time.time() - val_start

    logger.info(
        f"[Validation END] loss={avg_loss:.4f} | accuracy={accuracy:.4f} | "
        f"{total_samples} samples, {num_batches} batches | {val_time:.1f}s"
    )
    logger.info("-" * 45)

    logger.info("Saving final baseline checkpoint...")
    checkpoint_manager.save_baseline(
        step=step,
        model=model,
        accuracy=accuracy,
    )

if __name__ == "__main__":
    main()
