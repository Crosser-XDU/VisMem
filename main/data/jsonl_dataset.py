from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from torch.utils.data import Dataset

@dataclass
class Sample:
    id: str
    image: Optional[str]
    prompt: str
    answer: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class JsonlVLDataset(Dataset):
    def __init__(self, jsonl_path: str):
        self.items: List[Sample] = []
        root_dir = os.path.dirname(os.path.abspath(jsonl_path))
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                obj = json.loads(line)
                image = obj.get("image", None)
                if image is not None and not os.path.isabs(image):
                    image = os.path.join(root_dir, image)
                self.items.append(
                    Sample(
                        id=str(obj.get("id", len(self.items))),
                        image=image,
                        prompt=obj["prompt"],
                        answer=obj.get("answer", None),
                        meta={k:v for k,v in obj.items() if k not in ("id","image","prompt","answer")}
                    )
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Sample:
        return self.items[idx]
