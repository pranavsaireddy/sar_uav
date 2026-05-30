"""
SAR UAV Detection System - Inference Engine

Wraps the trained model for production inference.
Handles image preprocessing, NMS, and DetectionResult construction.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from backend.models.fusion_model import SARFusionModel, build_model


# ─── NMS ────────────────────────────────────────────────────────────────────────

def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
    """Non-maximum suppression. Returns kept indices."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    kept = []

    while len(order) > 0:
        i = order[0]
        kept.append(int(i))
        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-8)
        order = order[1:][iou <= iou_threshold]

    return kept


def decode_boxes(
    raw: torch.Tensor,           # (H, W, A, 5+C)
    conf_threshold: float = 0.45,
) -> list[dict]:
    """Decode raw YOLO output into list of detection dicts."""
    H, W, A, _ = raw.shape
    detections = []

    for gy in range(H):
        for gx in range(W):
            for a in range(A):
                obj = raw[gy, gx, a, 4].item()
                cls = raw[gy, gx, a, 5:].max().item()
                conf = obj * cls

                if conf < conf_threshold:
                    continue

                cx = (gx + raw[gy, gx, a, 0].item()) / W
                cy = (gy + raw[gy, gx, a, 1].item()) / H
                bw = raw[gy, gx, a, 2].item() / W
                bh = raw[gy, gx, a, 3].item() / H

                detections.append({
                    "cx": float(np.clip(cx, 0, 1)),
                    "cy": float(np.clip(cy, 0, 1)),
                    "w": float(np.clip(bw, 0, 1)),
                    "h": float(np.clip(bh, 0, 1)),
                    "confidence": float(conf),
                })

    if not detections:
        return []

    # NMS
    boxes_arr = np.array([[d["cx"], d["cy"], d["w"], d["h"]] for d in detections])
    scores_arr = np.array([d["confidence"] for d in detections])
    kept = nms(boxes_arr, scores_arr)
    return [detections[i] for i in kept]


# ─── Image Preprocessing ─────────────────────────────────────────────────────────

def preprocess_rgb(img: Image.Image, size: int = 320) -> torch.Tensor:
    """Returns (1, 3, size, size) float32 tensor in [0,1]."""
    img = img.convert("RGB").resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def preprocess_thermal(img: Image.Image, size: int = 320) -> torch.Tensor:
    """Returns (1, 1, size, size) float32 tensor in [0,1]. Handles 8 and 16 bit."""
    if img.mode in ("I", "I;16"):
        arr = np.array(img, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    else:
        arr = np.array(img.convert("L"), dtype=np.float32) / 255.0
    arr = np.array(Image.fromarray((arr * 255).astype(np.uint8)).resize((size, size)), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


# ─── Physics-Informed Survival Likelihood ────────────────────────────────────────

def compute_survival_likelihood(
    thermal: np.ndarray,      # (H, W) float32 in [0, 1]
    bboxes: list[dict],       # decoded bounding boxes (cx, cy, w, h normalised)
    confidence: float,
    consistency: float,
) -> float:
    """
    Estimate survival likelihood from observable image signals.

    Four components (each 0-1), combined with domain-informed weights:

    1. thermal_signal  — mean thermal intensity inside detected bbox(es)
                         relative to the image background.
                         A living body (~37°C) reads brighter than surroundings.

    2. thermal_contrast — (bbox mean − background mean) / (background std + ε)
                          Normalised z-score: how many std-devs above ambient.
                          High contrast = warm body against cool environment.

    3. confidence      — model detection confidence (already meaningful).

    4. posture_factor  — bounding box aspect ratio proxy.
                         h > w  → upright/standing  → higher mobility signal.
                         w > h  → prone/lying down  → lower mobility signal.
                         Neutral when no box detected.
    """
    H, W = thermal.shape

    if not bboxes:
        thermal_signal = 0.0
        thermal_contrast = 0.0
        posture_factor = 0.5
    else:
        # Pre-compute global stats once — person bbox is <1% of 320×320,
        # so global mean/std is a good background proxy without per-box masking.
        global_mean = float(thermal.mean())
        global_std  = float(thermal.std()) + 1e-6

        box_signals, box_contrasts, posture_scores = [], [], []

        for b in bboxes:
            cx, cy, bw, bh = b["cx"], b["cy"], b["w"], b["h"]
            x0 = max(0, int((cx - bw / 2) * W))
            y0 = max(0, int((cy - bh / 2) * H))
            x1 = min(W, int((cx + bw / 2) * W))
            y1 = min(H, int((cy + bh / 2) * H))

            if x1 <= x0 or y1 <= y0:
                continue

            roi_mean = float(thermal[y0:y1, x0:x1].mean())
            box_signals.append(roi_mean)
            box_contrasts.append((roi_mean - global_mean) / global_std)

            ratio = (y1 - y0) / (x1 - x0 + 1e-6)
            if ratio >= 1.2:
                posture_scores.append(0.85)   # upright → mobile
            elif ratio <= 0.7:
                posture_scores.append(0.30)   # prone → injured
            else:
                posture_scores.append(0.60)   # ambiguous

        thermal_signal   = float(np.mean(box_signals))   if box_signals   else 0.0
        raw_contrast     = float(np.mean(box_contrasts)) if box_contrasts else 0.0
        thermal_contrast = float(1 / (1 + np.exp(-raw_contrast)))
        posture_factor   = float(np.mean(posture_scores)) if posture_scores else 0.5

    score = (
        0.30 * thermal_signal    +
        0.25 * thermal_contrast  +
        0.25 * confidence        +
        0.10 * consistency       +
        0.10 * posture_factor
    )
    return float(np.clip(score, 0.0, 1.0))


# ─── Inference Engine ────────────────────────────────────────────────────────────

class InferenceEngine:
    """Thread-safe inference engine (use asyncio.Lock in async context)."""

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_mode: str = "full",
        img_size: int = 320,
        conf_threshold: float = 0.45,
        device: Optional[str] = None,
    ):
        self.img_size = img_size
        self.conf_threshold = conf_threshold
        self.frame_counter = 0

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = build_model(model_mode).to(self.device)
        self.model.eval()

        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state, strict=False)
            self.model = self.model.float()  # AMP saves fp16 weights; cast back to fp32 for inference
            print(f"Loaded checkpoint: {checkpoint_path}")
        else:
            print("No checkpoint found — using random weights (demo mode)")

        self.model_mode = model_mode

    @torch.no_grad()
    def infer(
        self,
        rgb_bytes: bytes,
        thermal_bytes: bytes,
        gps: Optional[dict] = None,
    ) -> dict:
        """
        Run inference on raw image bytes.
        Returns a DetectionResult-compatible dict.
        """
        t0 = time.perf_counter()
        self.frame_counter += 1

        rgb_img = Image.open(io.BytesIO(rgb_bytes))
        thm_img = Image.open(io.BytesIO(thermal_bytes))

        rgb_t = preprocess_rgb(rgb_img, self.img_size).to(self.device).float()
        thm_t = preprocess_thermal(thm_img, self.img_size).to(self.device).float()

        with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
            outputs = self.model(rgb_t, thm_t)

        # Decode boxes from first batch item
        raw_boxes = outputs["boxes"][0].cpu()  # (H, W, A, 5+C)
        bboxes = decode_boxes(raw_boxes, self.conf_threshold)

        confidence = float(outputs["confidence"][0].item())
        consistency = float(outputs["consistency_score"][0].item())
        weights = outputs["modality_weights"].cpu().tolist()

        # Physics-informed survival likelihood
        thm_np = thm_t[0, 0].cpu().numpy()  # (H, W) in [0,1]
        survival = compute_survival_likelihood(thm_np, bboxes, confidence, consistency)

        detected = len(bboxes) > 0 and confidence > self.conf_threshold

        # Build explanation
        if detected:
            if consistency > 0.65:
                explanation = "Detected human — RGB body shape aligned with thermal heat signature."
            elif consistency > 0.4:
                explanation = "Possible survivor detected — moderate cross-modal agreement."
            else:
                explanation = "Shape detected but low thermal alignment — verify manually."
        else:
            if confidence > 0.2:
                explanation = "Thermal anomaly detected — suppressed (no RGB body shape match)."
            else:
                explanation = "No survivor detected in this frame."

        inference_ms = (time.perf_counter() - t0) * 1000

        return {
            "detected": detected,
            "confidence": round(confidence, 4),
            "consistency_score": round(consistency, 4),
            "survival_likelihood": round(survival, 4),
            "bounding_boxes": bboxes,
            "gps_location": gps or {"lat": 17.385, "lon": 78.487, "altitude": 50.0},
            "explanation": explanation,
            "frame_id": f"F{self.frame_counter:06d}",
            "inference_ms": round(inference_ms, 2),
            "modality_weights": {"rgb": round(weights[0], 4), "thermal": round(weights[1], 4)},
        }

    @property
    def is_ready(self) -> bool:
        return self.model is not None
