"""
SAR UAV Detection System - Dataset

Supports:
- LLVIP paired RGB + infrared images
- KAIST multispectral pedestrian dataset
- SAR JSON format (prepared labels)
- Synthetic fallback (via simulation module)

Synchronized augmentation applied identically to both modalities.
"""

import json
import os
import random
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset, DataLoader


# ─── Label helpers ─────────────────────────────────────────────────────────────

def parse_yolo_label(txt_path: Path) -> list[list[float]]:
    """Returns list of [cx, cy, w, h] normalized boxes."""
    boxes = []
    if not txt_path.exists():
        return boxes
    with open(txt_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                boxes.append([float(p) for p in parts[1:5]])
    return boxes


# ─── Synchronized augmentation ─────────────────────────────────────────────────

class SyncAugment:
    """
    Apply identical spatial transforms to RGB and thermal.
    Color/noise perturbations are modality-specific.
    """

    def __init__(self, size: int = 320, is_train: bool = True):
        self.size = size
        self.is_train = is_train

    def __call__(
        self,
        rgb: np.ndarray,         # (H, W, 3) uint8
        thermal: np.ndarray,     # (H, W) uint8
        boxes: list[list[float]],
    ) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
        # Resize both
        rgb = cv2.resize(rgb, (self.size, self.size))
        thermal = cv2.resize(thermal, (self.size, self.size))

        if not self.is_train:
            return rgb, thermal, boxes

        # Horizontal flip (p=0.5)
        if random.random() < 0.5:
            rgb = cv2.flip(rgb, 1)
            thermal = cv2.flip(thermal, 1)
            boxes = [[1.0 - cx, cy, w, h] for cx, cy, w, h in boxes]

        # Vertical flip (p=0.5)
        if random.random() < 0.5:
            rgb = cv2.flip(rgb, 0)
            thermal = cv2.flip(thermal, 0)
            boxes = [[cx, 1.0 - cy, w, h] for cx, cy, w, h in boxes]

        # RGB color jitter (RGB only)
        if random.random() < 0.7:
            brightness = 1.0 + random.uniform(-0.3, 0.3)
            contrast = 1.0 + random.uniform(-0.3, 0.3)
            rgb = np.clip(rgb.astype(np.float32) * brightness * contrast, 0, 255).astype(np.uint8)

        # Thermal noise (IR sensor variance)
        if random.random() < 0.5:
            noise = np.random.normal(0, 0.02 * 255, thermal.shape).astype(np.float32)
            thermal = np.clip(thermal.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Random crop + resize (p=0.3)
        if random.random() < 0.3 and boxes:
            scale = random.uniform(0.7, 1.0)
            h, w = self.size, self.size
            new_h, new_w = int(h * scale), int(w * scale)
            y0 = random.randint(0, h - new_h)
            x0 = random.randint(0, w - new_w)

            rgb = cv2.resize(rgb[y0:y0+new_h, x0:x0+new_w], (self.size, self.size))
            thermal = cv2.resize(thermal[y0:y0+new_h, x0:x0+new_w], (self.size, self.size))

            # Adjust boxes
            new_boxes = []
            for cx, cy, bw, bh in boxes:
                ncx = (cx * w - x0) / new_w
                ncy = (cy * h - y0) / new_h
                nw = bw / scale
                nh = bh / scale
                if 0.0 < ncx < 1.0 and 0.0 < ncy < 1.0:
                    new_boxes.append([
                        np.clip(ncx, 0, 1), np.clip(ncy, 0, 1),
                        np.clip(nw, 0, 1), np.clip(nh, 0, 1),
                    ])
            boxes = new_boxes

        return rgb, thermal, boxes


# ─── Dataset classes ────────────────────────────────────────────────────────────

class LLVIPDataset(Dataset):
    """
    Loads paired RGB + infrared images from LLVIP structure:
      LLVIP/
        visible/{train,test}/
        infrared/{train,test}/
        Annotations/YOLO_Format/{train,test}/
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        img_size: int = 320,
    ):
        self.root = Path(root)
        self.split = split
        self.augment = SyncAugment(img_size, is_train=(split == "train"))

        rgb_dir = self.root / "visible" / split
        ir_dir = self.root / "infrared" / split
        label_dir = self.root / "Annotations" / "YOLO_Format" / split

        rgb_files = sorted(rgb_dir.glob("*.jpg")) + sorted(rgb_dir.glob("*.png"))
        self.samples = []
        for rgb_path in rgb_files:
            ir_path = ir_dir / rgb_path.name
            if not ir_path.exists():
                ir_path = ir_dir / (rgb_path.stem + ".png")
            label_path = label_dir / (rgb_path.stem + ".txt")
            if ir_path.exists():
                self.samples.append((rgb_path, ir_path, label_path))

        print(f"LLVIP {split}: {len(self.samples)} pairs loaded")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        rgb_path, ir_path, label_path = self.samples[idx]

        # Load RGB
        rgb = cv2.imread(str(rgb_path))
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # Load thermal (handle 8-bit and 16-bit)
        thermal_img = Image.open(str(ir_path))
        if thermal_img.mode in ("I", "I;16"):
            thermal_arr = np.array(thermal_img, dtype=np.float32)
            thermal_arr = ((thermal_arr - thermal_arr.min()) /
                           (thermal_arr.max() - thermal_arr.min() + 1e-8) * 255).astype(np.uint8)
        else:
            thermal_arr = np.array(thermal_img.convert("L"))

        boxes = parse_yolo_label(label_path)

        # Augment
        rgb, thermal_arr, boxes = self.augment(rgb, thermal_arr, boxes)

        # To tensors [0,1]
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        thm_t = torch.from_numpy(thermal_arr).unsqueeze(0).float() / 255.0

        return {
            "rgb": rgb_t,
            "thermal": thm_t,
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            "labels": torch.zeros(len(boxes), dtype=torch.long),
            "image_id": rgb_path.stem,
        }


class SARJSONDataset(Dataset):
    """Loads from prepared SAR JSON label files."""

    def __init__(self, json_path: str, img_size: int = 320, split: str = "train"):
        with open(json_path) as f:
            self.records = json.load(f)
        self.split = split
        self.augment = SyncAugment(img_size, is_train=(split == "train"))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        rgb = cv2.cvtColor(cv2.imread(rec["rgb_path"]), cv2.COLOR_BGR2RGB)
        thermal = np.array(Image.open(rec["thermal_path"]).convert("L"))
        boxes = rec.get("boxes", [])

        rgb, thermal, boxes = self.augment(rgb, thermal, boxes)

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        thm_t = torch.from_numpy(thermal).unsqueeze(0).float() / 255.0

        return {
            "rgb": rgb_t,
            "thermal": thm_t,
            "boxes": torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros(0, 4),
            "labels": torch.zeros(len(boxes), dtype=torch.long),
            "image_id": rec.get("id", str(idx)),
        }


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate to handle variable-length box lists."""
    return {
        "rgb": torch.stack([b["rgb"] for b in batch]),
        "thermal": torch.stack([b["thermal"] for b in batch]),
        "boxes": [b["boxes"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "image_ids": [b["image_id"] for b in batch],
    }


def build_dataloader(
    root: str,
    split: str = "train",
    batch_size: int = 16,
    img_size: int = 320,
    num_workers: int = 4,
    dataset_type: str = "llvip",
) -> DataLoader:
    """Factory for DataLoader."""
    if dataset_type == "llvip":
        ds = LLVIPDataset(root, split, img_size)
    elif dataset_type == "json":
        ds = SARJSONDataset(root, img_size, split)
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        collate_fn=collate_fn,
        drop_last=(split == "train"),
    )
