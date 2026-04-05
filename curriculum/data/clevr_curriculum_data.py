# -*- coding: utf-8 -*-
"""
Encoding Process

    1. Build answer vocabulary from training question files.
    2. Create dataset using S3 CLEVR dataset.
    3. Wrap dataset in DataLoader using custom collate function.
    4. Iterate through DataLoader to generate encoded batches.
"""

import io
import json
from typing import Dict, List, Optional, Any

import boto3
import torch
from torch.utils.data import Dataset
from PIL import Image
from transformers import ViltProcessor

#Util functions to download files from S3 bucket
class S3Client:
    def __init__(self, bucket: str):
        self.bucket = bucket
        self.client = boto3.client("s3")

    def load_json(self, key: str) -> Any:
        """Download a JSON file from S3 and return it as a Python dict."""
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))

    def load_image(self, key: str) -> Image.Image:
        """Download an image from S3 and return it as a PIL Image."""
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")

def build_answer_vocab_s3(
    s3: S3Client,
    question_keys: List[str]
) -> Dict[str, int]:

    """
    Builds an answer vocabulary for CLEVR from question JSON files stored in S3.

    Returns:
        Dictionary mapping: answer string -> integer ID
    """
    answers = set()
    for key in question_keys:
        data = s3.load_json(key)["questions"]
        for q in data:
            if "answer" in q:
                answers.add(str(q["answer"]).strip().lower())
    answers = sorted(answers)
    return {a: i for i, a in enumerate(answers)}


# Dataset class that loads image + question and encodes using ViLT
class CLEVRCurriculumViltDatasetS3(Dataset):
    """
    PyTorch Dataset for CLEVR VQA using curriculum tiers.

    - Loads question JSON files from S3
    - Loads images from S3
    - Encodes image + question using ViLTProcessor
    - Converts answers to label IDs 

    Supports curriculum learning by allowing tier filtering.
    """
    def __init__(
        self,
        bucket: str,
        images_prefix: str,      
        questions_prefix: str,    
        processor: ViltProcessor,
        split: str, # train, val, or test
        answer2id: Optional[Dict[str, int]] = None, # mapping from answer string to integer ID
        tiers: Optional[List[int]] = None, # curriculum tiers to use
        max_length: int = 32, # max length of tokenized questions
    ):
        assert split in {"train", "val", "test"}

        # Initialize S3 client
        self.s3 = S3Client(bucket)

        self.images_prefix = images_prefix
        self.questions_prefix = questions_prefix
        self.processor = processor
        self.split = split
        self.answer2id = answer2id
        self.max_length = max_length

        self.samples: List[Dict[str, Any]] = []

        # Load questions from S3
        if split in {"train", "val"} and tiers is not None:
            for t in tiers:
                # Construct S3 key for tier-specific question file
                qkey = f"{questions_prefix}/CLEVR_{split}_questions_L{t}.json"

                questions = self.s3.load_json(qkey)["questions"]

                for q in questions:
                    self.samples.append({
                        "question": q["question"],
                        "answer": q.get("answer"),
                        "image_filename": q["image_filename"],
                        "question_index": q.get("question_index", -1),
                        "tier": t, # curriculum tier
                    })
        else:
            # Load all questions for the split with no tier filtering
            qkey = f"{questions_prefix}/CLEVR_{split}_questions.json"
            questions = self.s3.load_json(qkey)["questions"]

            for q in questions:
                self.samples.append({
                    "question": q["question"],
                    "answer": q.get("answer"),
                    "image_filename": q["image_filename"],
                    "question_index": q.get("question_index", -1),
                    "tier": -1, # -1 means no tier filtering
                })

    def __len__(self):
        """
        Returns total number of samples in dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Loads one sample:
        - Downloads image from S3
        - Encodes image + question using ViLTProcessor
        - Converts answer to label ID
        """

        sample = self.samples[idx]

         # Construct S3 key for image
        image_key = (
            f"{self.images_prefix}/"
            f"{self.split if self.split != 'val' else 'val'}/"
            f"{sample['image_filename']}"
        )
        # Download image from S3
        image = self.s3.load_image(image_key)

        # Encode using ViLTProcessor
        # This converts image + text into tensors
        encode = self.processor(
            images=image,
            text=sample["question"],
            return_tensors="pt",
            padding="max_length", # pad short questions to max_length   
            truncation=True, # truncate long questions to max_length
            max_length=self.max_length,
        )

        # Remove batch dimension (processor returns batch size 1)
        item = {k: v.squeeze(0) for k, v in encode.items()}

        # Convert the answer string to a number so the model can learn from it
        if sample.get("answer") is not None and self.answer2id is not None:
            key = str(sample["answer"]).strip().lower()
            item["labels"] = torch.tensor(self.answer2id[key], dtype=torch.long)

        # Also store the tier and question ID so it can be tracked later
        item["tier"] = torch.tensor(sample["tier"], dtype=torch.long)
        item["question_id"] = torch.tensor(sample["question_index"], dtype=torch.long)
        return item

def vilt_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:

    """
    Custom collate function for DataLoader to combine individual samples to a batch.

    - Stacks tensors across the batch dimension
    - Handles optional 'labels' key separately

    Required because our dataset returns dictionaries.
    """
    out: Dict[str, torch.Tensor] = {}
    for k in batch[0].keys():
        if k == "labels":
            continue
        out[k] = torch.stack([b[k] for b in batch])
    if all("labels" in b for b in batch):
        out["labels"] = torch.stack([b["labels"] for b in batch])
    return out





