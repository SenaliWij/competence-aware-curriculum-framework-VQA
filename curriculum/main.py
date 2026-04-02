# main.py
"""
Entry point for curriculum-aware VQA training.

All configurations live here in one place. Running this file
will build datasets from S3, initialise the model and curriculum
components, optionally resume from a checkpoint, and start training.

Usage:
    python main.py
"""

import json
import logging
import os
import random
import numpy as np
import torch
from transformers import ViltProcessor

from data.clevr_dataset_py import (
    CLEVRCurriculumViltDatasetS3,
    vilt_collate_fn,
)
from models.vilt_adapter import ViLTAdapter
from services.checkpoint_service import CheckpointManager
from training.curriculum_trainer import CurriculumTrainer


# =====================================================================
# CONFIGURATION  — edit these values to match your setup
# =====================================================================

CONFIG = {
    # ---- General ----
    "seed":                42,

    # ---- S3 ----
    "s3_bucket":           "clevr-curriculum",
    "s3_images_prefix":    "dataset/images",
    "s3_questions_prefix": "dataset/questions",

    # ---- Answer vocab (local JSON) ----
    "answer_vocab_path":   os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "data", "answer_vocab.json"
    ),

    # ---- Dataset ----
    "tiers":               [1, 2, 3, 4, 5],
    "max_question_length": 32,

    # ---- Model ----
    "model_name":          "dandelin/vilt-b32-finetuned-vqa",
    "learning_rate":       5e-5,
    "freeze_backbone":     False,
    "device":              "cuda",   # "cuda" or "cpu"

    # ---- Curriculum ----
    "beta":                0.9,      # EMA smoothing factor
    "entropy_weight":      0.6,      # α  in C = α·(1−H/Hmax) + (1−α)·(1−L/L0)
    "loss_weight":         0.4,
    "c_min":               0.05,     # competence clamp lower
    "c_max":               0.99,     # competence clamp upper

    # ---- Tier difficulty exponents (power-law) ----
    "difficulty": {
        1: 1.0,   # Attribute & Existence  (easiest)
        2: 3.0,   # Counting / Compare Int
        3: 5.0,   # Compare Attribute
        4: 7.0,   # Relational Tasks
        5: 10.0,  # Complex Composition    (hardest)
    },

    # ---- Soft Self-Paced Learning ----
    "spl_lambda_init":     0.5,      # SPL temperature at low competence
    "spl_lambda_max":      5.0,      # SPL temperature at high competence

    # ---- Training loop ----
    "batch_size":          32,
    "num_steps":           10_000,
    "val_every":           1000,     # validate every N steps
    "checkpoint_every":    200,      # checkpoint to S3 every N steps
    "log_every":           50,       # print log every N steps

    # ---- Checkpointing ----
    "run_name":            "curriculum_run_01",
    "checkpoint_prefix":   "checkpoints",
    "resume":              True,     # try to resume from latest S3 checkpoint
}


# =====================================================================
# LOGGING
# =====================================================================

def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-28s  %(levelname)-7s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# =====================================================================
# LOAD ANSWER VOCABULARY
# =====================================================================

def load_answer_vocab(cfg: dict) -> dict:
    """
    Load the answer → id mapping from a local JSON file.
    Expected format:  {"yes": 27, "no": 20, ...}
    """
    path = cfg["answer_vocab_path"]
    with open(path, "r") as f:
        answer2id = json.load(f)
    logging.info("Loaded answer vocab from %s  (%d classes)", path, len(answer2id))
    return answer2id


# =====================================================================
# BUILD DATASETS
# =====================================================================

def build_datasets(cfg: dict, processor, answer2id: dict):
    """
    Create one training dataset per tier + one combined validation dataset.
    All data is streamed from S3.
    """
    tier_datasets = {}
    for t in cfg["tiers"]:
        tier_datasets[t] = CLEVRCurriculumViltDatasetS3(
            bucket=cfg["s3_bucket"],
            images_prefix=cfg["s3_images_prefix"],
            questions_prefix=cfg["s3_questions_prefix"],
            processor=processor,
            split="train",
            answer2id=answer2id,
            tiers=[t],
            max_length=cfg["max_question_length"],
        )
        logging.info("Tier %d training set: %d samples", t, len(tier_datasets[t]))

    val_dataset = CLEVRCurriculumViltDatasetS3(
        bucket=cfg["s3_bucket"],
        images_prefix=cfg["s3_images_prefix"],
        questions_prefix=cfg["s3_questions_prefix"],
        processor=processor,
        split="val",
        answer2id=answer2id,
        tiers=cfg["tiers"],
        max_length=cfg["max_question_length"],
    )
    logging.info("Validation set: %d samples", len(val_dataset))

    # Per-tier validation datasets
    tier_val_datasets = {}
    for t in cfg["tiers"]:
        tier_val_datasets[t] = CLEVRCurriculumViltDatasetS3(
            bucket=cfg["s3_bucket"],
            images_prefix=cfg["s3_images_prefix"],
            questions_prefix=cfg["s3_questions_prefix"],
            processor=processor,
            split="val",
            answer2id=answer2id,
            tiers=[t],
            max_length=cfg["max_question_length"],
        )
        logging.info("Tier %d validation set: %d samples", t, len(tier_val_datasets[t]))

    return tier_datasets, val_dataset, tier_val_datasets


# =====================================================================
# MAIN
# =====================================================================

def set_seed(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    cfg = CONFIG
    set_seed(cfg.get("seed", 42))
    _setup_logging()
    logger = logging.getLogger("main")

    logger.info("=" * 60)
    logger.info("  Competence-Aware Curriculum Training")
    logger.info("=" * 60)

    # ---- Processor & vocab ----
    processor = ViltProcessor.from_pretrained(cfg["model_name"])
    answer2id = load_answer_vocab(cfg)
    num_classes = len(answer2id)

    # ---- Datasets (S3) ----
    tier_datasets, val_dataset, tier_val_datasets = build_datasets(cfg, processor, answer2id)

    # ---- Model ----
    model = ViLTAdapter(
        model_name=cfg["model_name"],
        num_labels=num_classes,
        learning_rate=cfg["learning_rate"],
        device=cfg["device"],
        freeze_backbone=cfg["freeze_backbone"],
    )
    logger.info("Model loaded: %s  (num_labels=%d)", cfg["model_name"], num_classes)

    # ---- Checkpoint manager (S3) ----
    ckpt = CheckpointManager(
        bucket=cfg["s3_bucket"],
        run_name=cfg["run_name"],
        prefix=cfg["checkpoint_prefix"],
    )

     trainer = CurriculumTrainer(
        model=model,
        tier_datasets=tier_datasets,
        val_dataset=val_dataset,
        num_classes=num_classes,
        checkpoint_manager=ckpt,
        batch_size=cfg["batch_size"],
        num_steps=cfg["num_steps"],
        beta=cfg["beta"],
        entropy_weight=cfg["entropy_weight"],
        loss_weight=cfg["loss_weight"],
        collate_fn=vilt_collate_fn,
        difficulty=cfg["difficulty"],
        val_every=cfg["val_every"],
        checkpoint_every=cfg["checkpoint_every"],
        log_every=cfg["log_every"],
        tier_val_datasets=tier_val_datasets,
    )

    # ---- Resume from checkpoint if available ----
    start_step = 0
    if cfg["resume"]:
        start_step = trainer.resume(tag="latest")
        if start_step > 0:
            logger.info("Resumed from step %d", start_step)
        else:
            logger.info("No checkpoint found — starting from scratch")

    # ---- Train ----
    logger.info("Training for %d steps (starting at step %d)", cfg["num_steps"], start_step)
    history = trainer.train(start_step=start_step)

    logger.info("=" * 60)
    logger.info("  Training finished  |  %d steps completed", len(history))
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
