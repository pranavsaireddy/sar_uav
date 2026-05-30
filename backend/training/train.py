"""
SAR UAV Detection System - Training Loop

Features:
- Mixed precision (AMP) for RTX 4060 Ti
- Cosine LR schedule with linear warmup
- Hard negative mining (pool of 512 FP examples)
- Checkpoint saving (best val loss)
- TensorBoard / console logging
"""

from __future__ import annotations

import argparse
import math
import os
import random
import time
from collections import deque
from pathlib import Path

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.models.fusion_model import build_model
from backend.training.losses import SARLoss
from backend.training.dataset import build_dataloader


# ─── Hard Negative Pool ─────────────────────────────────────────────────────────

class HardNegativePool:
    """
    Maintains top-K hard false positives (model confident but no ground truth).
    Injects 20% hard negatives into batches every 10 steps.
    """

    def __init__(self, pool_size: int = 512):
        self.pool: deque = deque(maxlen=pool_size)

    def add(self, rgb: torch.Tensor, thermal: torch.Tensor, confidence: float):
        """Add a sample to the pool (stored on CPU)."""
        self.pool.append({
            "rgb": rgb.cpu(),
            "thermal": thermal.cpu(),
            "confidence": confidence,
        })

    def sample(self, n: int, device: torch.device) -> list[dict] | None:
        if len(self.pool) < n:
            return None
        # Sort by confidence descending (hardest first)
        items = sorted(self.pool, key=lambda x: x["confidence"], reverse=True)
        chosen = items[:n]
        return [{"rgb": c["rgb"].to(device), "thermal": c["thermal"].to(device)} for c in chosen]

    def __len__(self) -> int:
        return len(self.pool)


# ─── LR Schedule ────────────────────────────────────────────────────────────────

def get_lr(step: int, warmup_steps: int, total_steps: int, base_lr: float) -> float:
    """Linear warmup then cosine annealing."""
    if step < warmup_steps:
        return base_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ─── Evaluation ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, val_loader, criterion, device, amp: bool) -> dict:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in val_loader:
        rgb = batch["rgb"].to(device)
        thermal = batch["thermal"].to(device)
        targets = [
            {"boxes": b.to(device), "labels": l.to(device)}
            for b, l in zip(batch["boxes"], batch["labels"])
        ]

        with autocast("cuda", enabled=amp):
            preds = model(rgb, thermal)
            losses = criterion(preds, targets)

        total_loss += losses["total"].item()

        # Simple accuracy: did model predict detection correctly?
        gt_has_human = torch.tensor(
            [1.0 if t["boxes"].shape[0] > 0 else 0.0 for t in targets], device=device
        )
        pred_detected = (preds["confidence"] > 0.45).float()
        total_correct += (pred_detected == gt_has_human).sum().item()
        total_samples += len(targets)

    model.train()
    return {
        "loss": total_loss / max(1, len(val_loader)),
        "accuracy": total_correct / max(1, total_samples),
    }


# ─── Main Training ───────────────────────────────────────────────────────────────

def train(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Build model
    model = build_model(args.model).to(device)
    print(f"Parameters: {model.num_parameters:,}")

    # Data
    train_loader = build_dataloader(
        args.data, split="train",
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.workers,
        dataset_type=args.dataset_type,
    )
    val_loader = build_dataloader(
        args.data, split="test",
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.workers,
        dataset_type=args.dataset_type,
    )

    # Loss
    criterion = SARLoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # AMP scaler
    scaler = GradScaler("cuda", enabled=args.amp)

    # LR schedule params
    steps_per_epoch = len(train_loader)
    total_steps = args.epochs * steps_per_epoch
    warmup_steps = 5 * steps_per_epoch

    # Hard negative pool
    hn_pool = HardNegativePool(pool_size=512)

    # Checkpoint dir
    ckpt_dir = Path(args.ckpt_dir) if args.ckpt_dir else Path(args.data).parent / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")
    global_step = 0
    start_epoch = 1

    # Resume from checkpoint if requested
    if args.resume and Path(args.resume).exists():
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        # Periodic checkpoints save only state_dict; best saves full dict
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_val_loss = ckpt.get("val_loss", float("inf"))
        else:
            model.load_state_dict(ckpt)
            # Periodic checkpoint filename encodes epoch: sar_epoch_040.pt
            stem = Path(args.resume).stem  # e.g. sar_epoch_040
            parts = stem.split("_")
            if parts[-1].isdigit():
                start_epoch = int(parts[-1]) + 1
        model = model.float()
        # Advance global_step to match where LR schedule should be
        global_step = (start_epoch - 1) * steps_per_epoch
        print(f"Resuming from epoch {start_epoch}, global_step={global_step}")

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_losses: dict[str, float] = {}
        t0 = time.time()

        for step, batch in enumerate(train_loader):
            # LR update
            lr = get_lr(global_step, warmup_steps, total_steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            rgb = batch["rgb"].to(device)
            thermal = batch["thermal"].to(device)
            targets = [
                {"boxes": b.to(device), "labels": l.to(device)}
                for b, l in zip(batch["boxes"], batch["labels"])
            ]

            # Inject hard negatives every 10 batches (20% of batch)
            if global_step % 10 == 0 and len(hn_pool) >= 8:
                n_inject = max(1, args.batch_size // 5)
                hn_samples = hn_pool.sample(n_inject, device)
                if hn_samples:
                    hn_rgb = torch.stack([s["rgb"] for s in hn_samples])
                    hn_thm = torch.stack([s["thermal"] for s in hn_samples])
                    rgb = torch.cat([rgb, hn_rgb], dim=0)
                    thermal = torch.cat([thermal, hn_thm], dim=0)
                    targets += [{"boxes": torch.zeros(0, 4, device=device),
                                 "labels": torch.zeros(0, dtype=torch.long, device=device)}
                                for _ in hn_samples]

            optimizer.zero_grad()

            with autocast("cuda", enabled=args.amp):
                preds = model(rgb, thermal)
                losses = criterion(preds, targets)

            scaler.scale(losses["total"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()

            # Update hard negative pool
            with torch.no_grad():
                for b_idx, conf in enumerate(preds["confidence"][:args.batch_size]):
                    if conf.item() > 0.5 and targets[b_idx]["boxes"].shape[0] == 0:
                        hn_pool.add(
                            rgb[b_idx].detach(),
                            thermal[b_idx].detach(),
                            conf.item(),
                        )

            # Accumulate loss metrics
            for k, v in losses.items():
                epoch_losses[k] = epoch_losses.get(k, 0.0) + v.item()

            global_step += 1

            if step % 50 == 0:
                avg_total = epoch_losses.get("total", 0) / max(1, step + 1)
                print(
                    f"Epoch {epoch}/{args.epochs} | Step {step}/{steps_per_epoch} | "
                    f"Loss {avg_total:.4f} | LR {lr:.2e} | HN Pool {len(hn_pool)}"
                )

        # Epoch summary
        n_steps = max(1, steps_per_epoch)
        elapsed = time.time() - t0
        avg = {k: v / n_steps for k, v in epoch_losses.items()}
        print(
            f"\n[Epoch {epoch}] Loss={avg.get('total',0):.4f} | "
            f"Det={avg.get('detection',0):.4f} | "
            f"Cons={avg.get('consistency',0):.4f} | "
            f"Align={avg.get('alignment',0):.4f} | "
            f"Time={elapsed:.1f}s"
        )

        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device, args.amp)
        print(
            f"[Val] Loss={val_metrics['loss']:.4f} | "
            f"Accuracy={val_metrics['accuracy']:.2%}"
        )

        # Save best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt_path = ckpt_dir / "best_sar_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": best_val_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "model_mode": args.model,
                    "img_size": args.img_size,
                },
                ckpt_path,
            )
            print(f"  [BEST] Saved checkpoint -> {ckpt_path}")

        # Periodic checkpoint
        if epoch % 10 == 0:
            torch.save(
                model.state_dict(),
                ckpt_dir / f"sar_epoch_{epoch:03d}.pt",
            )

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


# ─── Entry Point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAR UAV Detection - Training")
    parser.add_argument("--data", type=str, required=True, help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--model", type=str, default="full", choices=["full", "edge"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--dataset-type", type=str, default="llvip", choices=["llvip", "json"])
    parser.add_argument("--ckpt-dir", type=str, default="", help="Checkpoint output dir (default: <data>/../checkpoints)")
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint to resume from")
    args = parser.parse_args()

    train(args)
