"""
SAR UAV Detection System - Loss Functions

Three jointly-trained losses:
1. DetectionLoss       - box regression + objectness (weight 1.0)
2. CrossModalConsistency - pushes feature alignment for true positives (weight 0.5)
3. FeatureAlignmentLoss - forces spatial activation alignment (weight 0.3)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    """
    YOLO-style detection loss.
    - Box regression: Smooth-L1 on (cx, cy, w, h)
    - Objectness: BCE at assigned anchor cells
    """

    def __init__(self, lambda_box: float = 5.0, lambda_obj: float = 1.0):
        super().__init__()
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj

    def forward(
        self,
        predictions: torch.Tensor,  # (B, H, W, A, 5+C)
        targets: list[dict],         # list of {boxes: (N,4), labels: (N,)}
    ) -> dict[str, torch.Tensor]:
        B, H, W, A, _ = predictions.shape
        device = predictions.device

        box_loss = torch.tensor(0.0, device=device)
        obj_loss = torch.tensor(0.0, device=device)
        n_pos = 0

        obj_targets = torch.zeros(B, H, W, A, device=device)

        for b, target in enumerate(targets):
            boxes = target.get("boxes", torch.empty(0, 4, device=device))
            if boxes.shape[0] == 0:
                continue

            for box in boxes:
                cx, cy, w, h = box
                # Map to grid cell
                gi = int(cx * W)
                gj = int(cy * H)
                gi = min(gi, W - 1)
                gj = min(gj, H - 1)

                # Assign to anchor 0 (simplified; full impl uses IoU matching)
                anchor_idx = 0
                obj_targets[b, gj, gi, anchor_idx] = 1.0

                # Box regression loss
                pred_box = predictions[b, gj, gi, anchor_idx, :4]
                target_box = box.to(device)
                box_loss = box_loss + F.smooth_l1_loss(pred_box, target_box)
                n_pos += 1

        if n_pos > 0:
            box_loss = box_loss / n_pos

        # Objectness BCE over all cells (cast to float32 — safe with AMP)
        pred_obj = predictions[..., 4]
        with torch.amp.autocast("cuda", enabled=False):
            obj_loss = F.binary_cross_entropy(pred_obj.float(), obj_targets.float())

        total = self.lambda_box * box_loss + self.lambda_obj * obj_loss
        return {
            "detection": total,
            "box": box_loss,
            "objectness": obj_loss,
        }


class CrossModalConsistencyLoss(nn.Module):
    """
    Pushes cosine similarity between pooled RGB and thermal features
    above a margin for positive samples (human present).
    Negative samples: no penalty (background can legitimately differ).

    Formula: L = max(0, margin - cosine_sim(pool(F_rgb), pool(F_thm))) for positives
    """

    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        rgb_features: torch.Tensor,   # (B, C, H, W)
        thm_features: torch.Tensor,   # (B, C, H, W)
        has_human: torch.Tensor,       # (B,) binary
    ) -> torch.Tensor:
        # Global average pooling
        rgb_pooled = rgb_features.flatten(2).mean(-1)  # (B, C)
        thm_pooled = thm_features.flatten(2).mean(-1)  # (B, C)

        cos_sim = F.cosine_similarity(rgb_pooled, thm_pooled, dim=1)  # (B,)

        # Only penalize positive samples
        pos_mask = has_human.float()
        loss = (F.relu(self.margin - cos_sim) * pos_mask).mean()
        return loss


class FeatureAlignmentLoss(nn.Module):
    """
    Forces both encoders to activate in the same spatial regions.
    Normalizes per-channel activation magnitude maps and computes MSE
    at feature map resolution (H/32 x W/32).
    """

    def forward(
        self,
        rgb_features: torch.Tensor,   # (B, C, H, W)
        thm_features: torch.Tensor,   # (B, C, H, W)
    ) -> torch.Tensor:
        # Per-channel activation magnitude (L2 across spatial)
        rgb_mag = rgb_features.pow(2).sum(1, keepdim=True).sqrt()  # (B,1,H,W)
        thm_mag = thm_features.pow(2).sum(1, keepdim=True).sqrt()  # (B,1,H,W)

        # Normalize to [0,1]
        def normalize(t: torch.Tensor) -> torch.Tensor:
            b_min = t.flatten(1).min(1)[0].view(-1, 1, 1, 1)
            b_max = t.flatten(1).max(1)[0].view(-1, 1, 1, 1)
            return (t - b_min) / (b_max - b_min + 1e-8)

        rgb_norm = normalize(rgb_mag)
        thm_norm = normalize(thm_mag)

        return F.mse_loss(rgb_norm, thm_norm)


class SARLoss(nn.Module):
    """Combined loss for joint training."""

    def __init__(
        self,
        w_detection: float = 1.0,
        w_consistency: float = 0.5,
        w_alignment: float = 0.3,
    ):
        super().__init__()
        self.detection_loss = DetectionLoss()
        self.consistency_loss = CrossModalConsistencyLoss()
        self.alignment_loss = FeatureAlignmentLoss()

        self.w_detection = w_detection
        self.w_consistency = w_consistency
        self.w_alignment = w_alignment

    def forward(
        self,
        predictions: dict,
        targets: list[dict],
    ) -> dict[str, torch.Tensor]:
        boxes = predictions["boxes"]
        rgb_feat = predictions["rgb_features"]
        thm_feat = predictions["thm_features"]

        # Has human flag from targets
        has_human = torch.tensor(
            [1.0 if t.get("boxes", torch.empty(0)).shape[0] > 0 else 0.0
             for t in targets],
            device=boxes.device,
        )

        det_losses = self.detection_loss(boxes, targets)
        cons_loss = self.consistency_loss(rgb_feat, thm_feat, has_human)
        align_loss = self.alignment_loss(rgb_feat, thm_feat)

        total = (
            self.w_detection * det_losses["detection"]
            + self.w_consistency * cons_loss
            + self.w_alignment * align_loss
        )

        return {
            "total": total,
            "detection": det_losses["detection"],
            "box": det_losses["box"],
            "objectness": det_losses["objectness"],
            "consistency": cons_loss,
            "alignment": align_loss,
        }
