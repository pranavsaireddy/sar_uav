"""
SAR Training Pipeline
Multimodal RGB-Thermal Detection Training with:
    - Detection loss (bbox + classification)
    - Cross-modal consistency loss
    - Feature alignment loss
    - Hard negative mining
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms as T
import torchvision.transforms.functional as TF

import numpy as np
import random
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from models.fusion_model import build_sar_model, SARFusionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("SAR-Train")


# ─────────────────────────────────────────────
# 1. DATASET
# ─────────────────────────────────────────────

@dataclass
class SARSample:
    rgb_path:     str
    thermal_path: str
    boxes:        List[List[float]]   # [cx, cy, w, h] normalized
    labels:       List[int]           # 0=human
    gps:          Optional[Dict]      = None


class SARDataset(Dataset):
    """
    Paired RGB + Thermal dataset for SAR detection.

    Expected directory structure:
        data/
            rgb/        *.jpg
            thermal/    *.png   (16-bit or 8-bit grayscale)
            labels/     *.json  {boxes: [...], labels: [...], gps: {...}}
    """

    IMG_SIZE = 640

    def __init__(self, data_dir: str, split: str = "train", augment: bool = True):
        self.data_dir = Path(data_dir)
        self.augment  = augment and (split == "train")
        self.samples  = self._load_samples(split)
        log.info(f"Dataset [{split}]: {len(self.samples)} samples loaded")

    def _load_samples(self, split: str) -> List[SARSample]:
        label_dir = self.data_dir / "labels"
        samples   = []
        for lf in sorted(label_dir.glob("*.json")):
            meta = json.loads(lf.read_text())
            if meta.get("split", "train") != split:
                continue
            samples.append(SARSample(
                rgb_path     = str(self.data_dir / "rgb" / (lf.stem + ".jpg")),
                thermal_path = str(self.data_dir / "thermal" / (lf.stem + ".png")),
                boxes        = meta["boxes"],
                labels       = meta["labels"],
                gps          = meta.get("gps"),
            ))
        return samples

    def _load_rgb(self, path: str) -> torch.Tensor:
        from PIL import Image
        img = Image.open(path).convert("RGB").resize((self.IMG_SIZE, self.IMG_SIZE))
        return TF.to_tensor(img)      # (3, H, W)

    def _load_thermal(self, path: str) -> torch.Tensor:
        from PIL import Image
        img = Image.open(path).convert("L").resize((self.IMG_SIZE, self.IMG_SIZE))
        t   = TF.to_tensor(img)      # (1, H, W)
        # Normalize to [0, 1] based on sensor range
        return t / t.max().clamp(min=1e-6)

    def _augment(
        self, rgb: torch.Tensor, thermal: torch.Tensor, boxes: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Synchronized augmentation — same transform applied to both modalities.
        """
        # Random horizontal flip
        if random.random() > 0.5:
            rgb     = TF.hflip(rgb)
            thermal = TF.hflip(thermal)
            if boxes.numel() > 0:
                boxes[:, 0] = 1.0 - boxes[:, 0]  # flip cx

        # Random vertical flip
        if random.random() > 0.5:
            rgb     = TF.vflip(rgb)
            thermal = TF.vflip(thermal)
            if boxes.numel() > 0:
                boxes[:, 1] = 1.0 - boxes[:, 1]

        # Color jitter (RGB only — thermal is physical)
        if random.random() > 0.3:
            rgb = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2)(rgb)

        # Thermal noise augmentation (simulate sensor noise)
        if random.random() > 0.4:
            noise   = torch.randn_like(thermal) * 0.02
            thermal = (thermal + noise).clamp(0, 1)

        # Mosaic / cutmix (simplified: random crop)
        if random.random() > 0.7:
            i, j, h, w = T.RandomCrop.get_params(
                rgb, output_size=(self.IMG_SIZE - 64, self.IMG_SIZE - 64)
            )
            rgb     = TF.resized_crop(rgb,     i, j, h, w, (self.IMG_SIZE, self.IMG_SIZE))
            thermal = TF.resized_crop(thermal, i, j, h, w, (self.IMG_SIZE, self.IMG_SIZE))

        return rgb, thermal, boxes

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        rgb     = self._load_rgb(s.rgb_path)
        thermal = self._load_thermal(s.thermal_path)

        boxes  = torch.tensor(s.boxes,  dtype=torch.float32)  if s.boxes  else torch.zeros(0, 4)
        labels = torch.tensor(s.labels, dtype=torch.long)     if s.labels else torch.zeros(0, dtype=torch.long)

        if self.augment:
            rgb, thermal, boxes = self._augment(rgb, thermal, boxes)

        return {
            "rgb":     rgb,
            "thermal": thermal,
            "boxes":   boxes,
            "labels":  labels,
            "gps":     s.gps or {},
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """Variable-length boxes require list collation."""
    return {
        "rgb":     torch.stack([b["rgb"]     for b in batch]),
        "thermal": torch.stack([b["thermal"] for b in batch]),
        "boxes":   [b["boxes"]  for b in batch],
        "labels":  [b["labels"] for b in batch],
        "gps":     [b["gps"]    for b in batch],
    }


# ─────────────────────────────────────────────
# 2. LOSS FUNCTIONS
# ─────────────────────────────────────────────

class DetectionLoss(nn.Module):
    """YOLO-style detection loss: bbox regression + classification."""

    def __init__(self, lambda_box: float = 5.0, lambda_obj: float = 1.0):
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.bce        = nn.BCELoss()

    def forward(
        self,
        pred_boxes: torch.Tensor,
        pred_obj:   torch.Tensor,
        target_boxes: List[torch.Tensor],
        target_labels: List[torch.Tensor],
    ) -> torch.Tensor:
        B = pred_boxes.shape[0]
        box_loss = torch.tensor(0.0, device=pred_boxes.device)
        obj_loss = torch.tensor(0.0, device=pred_boxes.device)

        for i in range(B):
            t_boxes  = target_boxes[i]
            t_labels = target_labels[i]

            if len(t_boxes) == 0:
                # No objects — objectness should be 0 everywhere
                obj_loss += self.bce(
                    pred_obj[i].flatten(),
                    torch.zeros_like(pred_obj[i].flatten())
                )
                continue

            # Simplified: match first anchor at grid cell (production uses IoU matching)
            for box, lbl in zip(t_boxes, t_labels):
                if lbl != 0:     # only human class
                    continue
                # Map normalized [cx,cy] to grid cell
                H, W = pred_boxes.shape[1], pred_boxes.shape[2]
                gx = int(box[0] * W)
                gy = int(box[1] * H)
                gx = max(0, min(gx, W - 1))
                gy = max(0, min(gy, H - 1))

                # Box regression (smooth L1)
                pred = pred_boxes[i, gy, gx, 0, :]
                box_loss += F.smooth_l1_loss(pred, box.to(pred.device))

                # Objectness at assigned cell
                obj_target = torch.zeros_like(pred_obj[i])
                obj_target[gy, gx, 0, 0] = 1.0
                obj_loss += self.bce(pred_obj[i], obj_target)

        return self.lambda_box * box_loss / B + self.lambda_obj * obj_loss / B


class CrossModalConsistencyLoss(nn.Module):
    """
    Penalises high detection confidence when RGB and Thermal features
    are inconsistent — the core false-positive reduction mechanism.
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        f_rgb: torch.Tensor,
        f_thm: torch.Tensor,
        has_human: torch.Tensor,    # (B,) bool
    ) -> torch.Tensor:
        # Cosine similarity between pooled features
        r = F.adaptive_avg_pool2d(f_rgb, 1).flatten(1)
        t = F.adaptive_avg_pool2d(f_thm, 1).flatten(1)
        sim = F.cosine_similarity(r, t, dim=1)   # (B,)

        # Positive pairs (human present) should be similar
        # Negative pairs (no human) can differ — no penalty
        pos_loss = F.relu(self.margin - sim[has_human]).mean()   if has_human.any()  else torch.tensor(0.0)
        return pos_loss


class FeatureAlignmentLoss(nn.Module):
    """
    Spatial feature alignment — RGB and Thermal should activate
    in the same regions when a human is present.
    """

    def forward(
        self,
        f_rgb: torch.Tensor,
        f_thm: torch.Tensor,
    ) -> torch.Tensor:
        # Channel-wise normalized activation maps
        r_map = F.normalize(f_rgb.abs().mean(dim=1, keepdim=True), dim=(2, 3))
        t_map = F.normalize(f_thm.abs().mean(dim=1, keepdim=True), dim=(2, 3))
        return F.mse_loss(r_map, t_map)


class SARLoss(nn.Module):
    """Composite loss combining all three objectives."""

    def __init__(
        self,
        w_det: float   = 1.0,
        w_consist: float = 0.5,
        w_align: float  = 0.3,
    ):
        super().__init__()
        self.det_loss   = DetectionLoss()
        self.cons_loss  = CrossModalConsistencyLoss()
        self.align_loss = FeatureAlignmentLoss()
        self.w_det      = w_det
        self.w_consist  = w_consist
        self.w_align    = w_align

    def forward(
        self,
        outputs:       Dict[str, torch.Tensor],
        target_boxes:  List[torch.Tensor],
        target_labels: List[torch.Tensor],
        f_rgb:         torch.Tensor,
        f_thm:         torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        has_human = torch.tensor(
            [len(lbl) > 0 and (lbl == 0).any() for lbl in target_labels],
            dtype=torch.bool,
        )

        det   = self.det_loss(outputs["boxes"], outputs["objectness"], target_boxes, target_labels)
        cons  = self.cons_loss(f_rgb, f_thm, has_human)
        align = self.align_loss(f_rgb, f_thm)

        total = self.w_det * det + self.w_consist * cons + self.w_align * align

        return {
            "total":       total,
            "detection":   det,
            "consistency": cons,
            "alignment":   align,
        }


# ─────────────────────────────────────────────
# 3. HARD NEGATIVE MINING
# ─────────────────────────────────────────────

class HardNegativeMiner:
    """
    Maintains a pool of high-confidence false-positive samples.
    These are replayed in future batches to push the model to discriminate.
    """

    def __init__(self, pool_size: int = 512, replay_ratio: float = 0.2):
        self.pool_size    = pool_size
        self.replay_ratio = replay_ratio
        self.pool: List[Dict] = []

    def update(self, samples: List[Dict], pred_confidences: List[float]):
        """Add high-confidence false positives to pool."""
        for sample, conf in zip(samples, pred_confidences):
            has_human = len(sample["labels"]) > 0
            if not has_human and conf > 0.5:
                # False positive — add to pool
                self.pool.append({"sample": sample, "conf": conf})

        # Keep top-k hardest
        self.pool = sorted(self.pool, key=lambda x: -x["conf"])[:self.pool_size]

    def sample_replay(self, n: int) -> List[Dict]:
        if not self.pool:
            return []
        k = min(n, len(self.pool))
        return [x["sample"] for x in random.sample(self.pool, k)]


# ─────────────────────────────────────────────
# 4. TRAINER
# ─────────────────────────────────────────────

@dataclass
class TrainConfig:
    data_dir:    str   = "data/"
    output_dir:  str   = "checkpoints/"
    epochs:      int   = 50
    batch_size:  int   = 8
    lr:          float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    model_mode:  str   = "full"      # full | edge
    amp:         bool  = True        # mixed precision
    workers:     int   = 4
    val_every:   int   = 5


class SARTrainer:

    def __init__(self, cfg: TrainConfig):
        self.cfg    = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"Training on: {self.device}")

        self.model     = build_sar_model(cfg.model_mode).to(self.device)
        self.loss_fn   = SARLoss().to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=cfg.epochs - cfg.warmup_epochs
        )
        self.scaler  = torch.cuda.amp.GradScaler(enabled=cfg.amp)
        self.miner   = HardNegativeMiner()
        self.best_map = 0.0

        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    def _get_loaders(self):
        train_ds = SARDataset(self.cfg.data_dir, "train", augment=True)
        val_ds   = SARDataset(self.cfg.data_dir, "val",   augment=False)

        train_loader = DataLoader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.cfg.workers, collate_fn=collate_fn, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.cfg.workers, collate_fn=collate_fn,
        )
        return train_loader, val_loader

    def _warmup_lr(self, epoch: int):
        if epoch < self.cfg.warmup_epochs:
            lr = self.cfg.lr * (epoch + 1) / self.cfg.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

    def _train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        metrics = {"total": 0, "detection": 0, "consistency": 0, "alignment": 0}

        for batch_idx, batch in enumerate(loader):
            rgb     = batch["rgb"].to(self.device)
            thermal = batch["thermal"].to(self.device)
            t_boxes = [b.to(self.device) for b in batch["boxes"]]
            t_lbls  = [l.to(self.device) for l in batch["labels"]]

            with torch.cuda.amp.autocast(enabled=self.cfg.amp):
                # Extract features separately for loss computation
                f_rgb  = self.model.rgb_encoder(rgb)
                f_thm  = self.model.thm_encoder(thermal)
                fused  = self.model.fusion(f_rgb, f_thm)
                filtered, consistency = self.model.anomaly(f_rgb, f_thm, fused)
                outputs = self.model.det_head(filtered)
                outputs["consistency_score"] = consistency

                losses = self.loss_fn(outputs, t_boxes, t_lbls, f_rgb, f_thm)

            self.optimizer.zero_grad()
            self.scaler.scale(losses["total"]).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Hard negative mining
            confs = outputs["confidence"].detach().flatten(1).max(1).values.cpu().tolist()
            self.miner.update(
                [{"labels": l.cpu(), "boxes": b.cpu()} for l, b in zip(batch["labels"], batch["boxes"])],
                confs,
            )

            for k in metrics:
                metrics[k] += losses[k].item()

            if batch_idx % 20 == 0:
                log.info(
                    f"Epoch {epoch} [{batch_idx}/{len(loader)}] "
                    f"Loss: {losses['total']:.4f} | Det: {losses['detection']:.4f} | "
                    f"Cons: {losses['consistency']:.4f}"
                )

        return {k: v / len(loader) for k, v in metrics.items()}

    @torch.no_grad()
    def _validate(self, loader: DataLoader) -> float:
        self.model.eval()
        total_correct = 0
        total_samples = 0

        for batch in loader:
            rgb     = batch["rgb"].to(self.device)
            thermal = batch["thermal"].to(self.device)
            outputs = self.model(rgb, thermal)

            conf = outputs["confidence"].flatten(1).max(1).values
            pred_pos = (conf > 0.5).cpu()
            true_pos = torch.tensor([len(l) > 0 for l in batch["labels"]])

            total_correct += (pred_pos == true_pos).sum().item()
            total_samples += len(true_pos)

        return total_correct / max(total_samples, 1)

    def train(self):
        train_loader, val_loader = self._get_loaders()
        log.info(f"Starting training | Epochs: {self.cfg.epochs} | Device: {self.device}")

        for epoch in range(1, self.cfg.epochs + 1):
            self._warmup_lr(epoch)
            train_metrics = self._train_epoch(train_loader, epoch)

            if epoch >= self.cfg.warmup_epochs:
                self.scheduler.step()

            log.info(
                f"Epoch {epoch} | "
                f"Train Loss: {train_metrics['total']:.4f} | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
            )

            if epoch % self.cfg.val_every == 0:
                acc = self._validate(val_loader)
                log.info(f"Validation Accuracy: {acc:.4f}")

                if acc > self.best_map:
                    self.best_map = acc
                    ckpt_path = Path(self.cfg.output_dir) / "best_sar_model.pt"
                    torch.save({
                        "epoch":       epoch,
                        "model_state": self.model.state_dict(),
                        "optim_state": self.optimizer.state_dict(),
                        "accuracy":    acc,
                        "config":      vars(self.cfg),
                    }, ckpt_path)
                    log.info(f"Saved best model → {ckpt_path}")

        log.info(f"Training complete | Best accuracy: {self.best_map:.4f}")


# ─────────────────────────────────────────────
# 5. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAR Multimodal Training")
    parser.add_argument("--data",       default="data/",        help="Dataset root")
    parser.add_argument("--output",     default="checkpoints/", help="Checkpoint dir")
    parser.add_argument("--epochs",     type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr",         type=float, default=3e-4)
    parser.add_argument("--model",      default="full", choices=["full", "edge"])
    args = parser.parse_args()

    cfg = TrainConfig(
        data_dir   = args.data,
        output_dir = args.output,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        model_mode = args.model,
    )
    trainer = SARTrainer(cfg)
    trainer.train()