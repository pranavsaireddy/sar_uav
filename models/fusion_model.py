"""
SAR Multimodal Fusion Model
RGB-Thermal Cross-Modal Detection for UAV-based Search and Rescue

Architecture:
    - Dual-stream encoders (RGB + Thermal)
    - Cross-modal transformer fusion
    - Anomaly filtering layer
    - Detection head with confidence scoring
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from typing import Optional, Tuple, Dict


# ─────────────────────────────────────────────
# 1. SHARED BACKBONE ENCODER
# ─────────────────────────────────────────────

class ModalityEncoder(nn.Module):
    """
    Lightweight EfficientNet-B0 backbone for either RGB or Thermal input.
    Outputs a spatial feature map: (B, 320, H/32, W/32)
    """

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        if pretrained and in_channels == 3:
            base = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        else:
            base = efficientnet_b0(weights=None)

        # If thermal (1-channel), adapt first conv
        if in_channels == 1:
            old_conv = base.features[0][0]
            base.features[0][0] = nn.Conv2d(
                1, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            # Initialize from mean of RGB weights
            with torch.no_grad():
                base.features[0][0].weight = nn.Parameter(
                    old_conv.weight.mean(dim=1, keepdim=True)
                )

        # Strip classifier, keep feature extractor
        self.features = base.features          # outputs (B, 1280, H/32, W/32)
        self.project = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.project(feat)   # (B, 256, H/32, W/32)


# ─────────────────────────────────────────────
# 2. CROSS-MODAL ATTENTION FUSION
# ─────────────────────────────────────────────

class CrossModalAttention(nn.Module):
    """
    Transformer-style cross-attention between RGB and Thermal feature maps.

    Key insight:
        - Query from one modality attends to keys/values of the other
        - Bidirectional: RGB→Thermal AND Thermal→RGB
        - Fused representation captures aligned cues only
    """

    def __init__(self, dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Projections for RGB attending to Thermal
        self.q_rgb = nn.Linear(dim, dim)
        self.k_thm = nn.Linear(dim, dim)
        self.v_thm = nn.Linear(dim, dim)

        # Projections for Thermal attending to RGB
        self.q_thm = nn.Linear(dim, dim)
        self.k_rgb = nn.Linear(dim, dim)
        self.v_rgb = nn.Linear(dim, dim)

        self.out_rgb = nn.Linear(dim, dim)
        self.out_thm = nn.Linear(dim, dim)

        self.norm_rgb = nn.LayerNorm(dim)
        self.norm_thm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def _attend(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, N, _ = q.shape
        H = self.num_heads

        q = q.reshape(B, N, H, self.head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B, -1, H, self.head_dim).permute(0, 2, 1, 3)
        v = v.reshape(B, -1, H, self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).permute(0, 2, 1, 3).reshape(B, N, self.dim)
        return out

    def forward(
        self,
        f_rgb: torch.Tensor,
        f_thm: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, C, H, W = f_rgb.shape
        N = H * W

        # Flatten spatial dims → sequence
        r = f_rgb.flatten(2).permute(0, 2, 1)   # (B, N, C)
        t = f_thm.flatten(2).permute(0, 2, 1)   # (B, N, C)

        # RGB queries attend to Thermal keys
        r_cross = self._attend(self.q_rgb(r), self.k_thm(t), self.v_thm(t))
        r_out   = self.norm_rgb(r + self.out_rgb(r_cross))

        # Thermal queries attend to RGB keys
        t_cross = self._attend(self.q_thm(t), self.k_rgb(r), self.v_rgb(r))
        t_out   = self.norm_thm(t + self.out_thm(t_cross))

        # Reshape back to spatial maps
        r_out = r_out.permute(0, 2, 1).reshape(B, C, H, W)
        t_out = t_out.permute(0, 2, 1).reshape(B, C, H, W)
        return r_out, t_out


class FusionModule(nn.Module):
    """
    Stacks N cross-modal attention layers, then merges into a single
    fused feature map via learned weighted combination.
    """

    def __init__(self, dim: int = 256, num_layers: int = 3, num_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossModalAttention(dim, num_heads) for _ in range(num_layers)
        ])
        # Learned modality weighting (softmax over 2 weights)
        self.modal_weight = nn.Parameter(torch.ones(2))

        self.ffn = nn.Sequential(
            nn.Conv2d(dim, dim * 2, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
        )

    def forward(self, f_rgb: torch.Tensor, f_thm: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            f_rgb, f_thm = layer(f_rgb, f_thm)

        # Weighted sum fusion
        w = F.softmax(self.modal_weight, dim=0)
        fused = w[0] * f_rgb + w[1] * f_thm
        return self.ffn(fused) + fused   # residual


# ─────────────────────────────────────────────
# 3. ANOMALY FILTERING LAYER
# ─────────────────────────────────────────────

class AnomalyFilter(nn.Module):
    """
    Rejects false positives via cross-modal consistency scoring.

    Logic:
        - If thermal is hot but RGB has no body shape → penalise
        - If RGB has human shape but thermal is cold → mark uncertain
        - Only "both aligned" passes with high confidence
    """

    def __init__(self, dim: int = 256):
        super().__init__()
        self.consistency_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(dim * 16, 128),
            nn.GELU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        # Per-pixel consistency map
        self.local_consistency = nn.Sequential(
            nn.Conv2d(dim * 2, 128, 3, padding=1, bias=False),
            nn.GELU(),
            nn.Conv2d(128, 1, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        f_rgb: torch.Tensor,
        f_thm: torch.Tensor,
        fused: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Global consistency score per image
        concat = torch.cat([f_rgb, f_thm], dim=1)
        global_score = self.consistency_head(concat)        # (B, 1)
        local_map    = self.local_consistency(concat)       # (B, 1, H, W)

        # Gate fused features with local consistency
        filtered = fused * local_map
        return filtered, global_score


# ─────────────────────────────────────────────
# 4. DETECTION HEAD
# ─────────────────────────────────────────────

class DetectionHead(nn.Module):
    """
    YOLO-style detection head producing:
        - Bounding boxes  (B, num_anchors, H, W, 4)   [cx, cy, w, h]
        - Objectness      (B, num_anchors, H, W, 1)
        - Class scores    (B, num_anchors, H, W, num_classes)
    """

    def __init__(
        self,
        in_dim: int = 256,
        num_anchors: int = 3,
        num_classes: int = 2,           # 0=human, 1=background
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        out_ch = num_anchors * (5 + num_classes)

        self.head = nn.Sequential(
            nn.Conv2d(in_dim, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Conv2d(256, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, out_ch, 1),
        )

        # Survival likelihood head (optional research output)
        self.survival_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, C, H, W = x.shape
        A = self.num_anchors
        K = self.num_classes

        raw = self.head(x)                                              # (B, A*(5+K), H, W)
        raw = raw.permute(0, 2, 3, 1).reshape(B, H, W, A, 5 + K)     # (B, H, W, A, 5+K)

        boxes      = raw[..., :4]               # raw offsets (to be decoded with anchors)
        objectness = torch.sigmoid(raw[..., 4:5])
        cls_scores = torch.sigmoid(raw[..., 5:])

        # Confidence = objectness × max class score
        confidence = objectness * cls_scores.max(dim=-1, keepdim=True).values

        survival = self.survival_head(x)         # (B, 1)

        return {
            "boxes":      boxes,
            "objectness": objectness,
            "cls_scores": cls_scores,
            "confidence": confidence,
            "survival":   survival,
        }


# ─────────────────────────────────────────────
# 5. FULL SAR FUSION MODEL
# ─────────────────────────────────────────────

class SARFusionModel(nn.Module):
    """
    Complete Multimodal RGB-Thermal SAR Detection Model

    Input:
        rgb   : (B, 3, H, W)   normalized [0,1]
        thermal: (B, 1, H, W)  normalized [0,1]

    Output:
        dict with detection outputs + consistency score + survival likelihood
    """

    def __init__(
        self,
        feature_dim: int     = 256,
        fusion_layers: int   = 3,
        num_heads: int       = 8,
        num_anchors: int     = 3,
        num_classes: int     = 2,
        pretrained: bool     = True,
    ):
        super().__init__()
        self.rgb_encoder  = ModalityEncoder(in_channels=3, pretrained=pretrained)
        self.thm_encoder  = ModalityEncoder(in_channels=1, pretrained=False)
        self.fusion       = FusionModule(feature_dim, fusion_layers, num_heads)
        self.anomaly      = AnomalyFilter(feature_dim)
        self.det_head     = DetectionHead(feature_dim, num_anchors, num_classes)

    def forward(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Feature extraction
        f_rgb = self.rgb_encoder(rgb)           # (B, 256, H/32, W/32)
        f_thm = self.thm_encoder(thermal)       # (B, 256, H/32, W/32)

        # Cross-modal fusion
        fused = self.fusion(f_rgb, f_thm)       # (B, 256, H/32, W/32)

        # Anomaly filtering
        filtered, consistency = self.anomaly(f_rgb, f_thm, fused)

        # Detection
        detections = self.det_head(filtered)
        detections["consistency_score"] = consistency

        return detections

    @torch.no_grad()
    def predict(
        self,
        rgb: torch.Tensor,
        thermal: torch.Tensor,
        conf_threshold: float = 0.5,
    ) -> Dict:
        """Inference-mode forward with NMS and structured output."""
        self.eval()
        out = self.forward(rgb, thermal)

        conf  = out["confidence"].squeeze()     # flatten anchors
        boxes = out["boxes"].squeeze()
        surv  = out["survival"].item()
        consistency = out["consistency_score"].item()

        # Simple threshold filtering (NMS would be applied in production)
        mask = conf.flatten() > conf_threshold

        return {
            "detected": bool(mask.any()),
            "confidence": float(conf.max()),
            "consistency_score": round(consistency, 4),
            "survival_likelihood": round(surv, 4),
            "bounding_boxes": boxes.flatten(0, -2)[mask.flatten()].tolist(),
            "explanation": _build_explanation(conf.max().item(), consistency),
        }


def _build_explanation(conf: float, consistency: float) -> str:
    if conf > 0.8 and consistency > 0.7:
        return "Detected human based on aligned RGB body structure and strong thermal signature."
    elif conf > 0.5 and consistency > 0.4:
        return "Probable human detection — moderate alignment between visual and thermal cues."
    elif conf > 0.5 and consistency < 0.3:
        return "Thermal hotspot detected but no matching body shape — likely false positive."
    else:
        return "No confident human detection in current frame."


# ─────────────────────────────────────────────
# 6. MODEL FACTORY
# ─────────────────────────────────────────────

def build_sar_model(mode: str = "full") -> SARFusionModel:
    """
    mode = 'full'       : full precision, training
    mode = 'edge'       : smaller model for UAV deployment
    mode = 'pretrained' : load with ImageNet weights
    """
    if mode == "edge":
        return SARFusionModel(
            feature_dim=128, fusion_layers=2, num_heads=4,
            num_anchors=3, pretrained=False,
        )
    return SARFusionModel(
        feature_dim=256, fusion_layers=3, num_heads=8,
        num_anchors=3, pretrained=(mode == "pretrained"),
    )


if __name__ == "__main__":
    model = build_sar_model("full")
    total = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"SAR Fusion Model | Parameters: {total:.1f}M")

    rgb     = torch.randn(2, 3, 640, 640)
    thermal = torch.randn(2, 1, 640, 640)
    out     = model(rgb, thermal)

    print("Output keys:", list(out.keys()))
    print("Confidence shape:", out["confidence"].shape)
    print("Survival:", out["survival"].shape)