import json
import io
import boto3
import torch
from typing import Dict, List, Optional, Any
from torch.utils.data import Dataset
from PIL import Image

def load_answer_vocab(cfg: dict) -> dict:
    path = cfg["answer_vocab_path"]
    with open(path, "r") as f:
        answer2id = json.load(f)
    print(f"Loaded answer vocab from {path}  ({len(answer2id)} classes)")
    return answer2id

class S3Client:
    def __init__(self, bucket: str):
        self.bucket = bucket
        self.client = boto3.client("s3")

    def load_json(self, key: str) -> Any:
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return json.loads(obj["Body"].read().decode("utf-8"))

    def load_image(self, key: str):
        obj = self.client.get_object(Bucket=self.bucket, Key=key)
        return Image.open(io.BytesIO(obj["Body"].read())).convert("RGB")

class CLEVRBaselineDatasetS3(Dataset):
    def __init__(
        self,
        bucket: str,
        images_prefix: str,      
        questions_prefix: str,    
        processor,
        filename: str,
        answer2id: Optional[Dict[str, int]] = None,
        max_length: int = 32,
        max_samples: Optional[int] = None,
    ):
        self.s3 = S3Client(bucket)
        self.images_prefix = images_prefix
        self.processor = processor
        self.answer2id = answer2id
        self.max_length = max_length
        self.samples = []
        
        qkey = f"{questions_prefix}/{filename}"
        print(f"Loading {qkey} from S3...")
        questions = self.s3.load_json(qkey)["questions"]
        print(f"Loaded {len(questions)} questions from {qkey}")
        
        for q in questions:
            self.samples.append({
                "question": q["question"],
                "answer": q.get("answer"),
                "image_filename": q["image_filename"],
                "question_index": q.get("question_index", -1),
            })
            
        if max_samples is not None and max_samples < len(self.samples):
            self.samples = self.samples[:max_samples]
            print(f"Truncated dataset to {max_samples} samples.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]
        split = s["image_filename"].split('_')[1] # 'train' or 'val'
        image_key = f"{self.images_prefix}/{split}/{s['image_filename']}"
        image = self.s3.load_image(image_key)

        encode = self.processor(
            images=image,
            text=s["question"],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        item = {k: v.squeeze(0) for k, v in encode.items()}
        if s.get("answer") is not None and self.answer2id is not None:
            key = str(s["answer"]).strip().lower()
            item["labels"] = torch.tensor(self.answer2id[key], dtype=torch.long)
        item["question_id"] = torch.tensor(s["question_index"], dtype=torch.long)
        return item

def vilt_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k in batch[0].keys():
        if k == "labels": continue
        out[k] = torch.stack([b[k] for b in batch])
    if all("labels" in b for b in batch):
        out["labels"] = torch.stack([b["labels"] for b in batch])
    return out
