"""
SAR UAV Detection System - Fusion Model
Multimodal RGB-Thermal cross-modal attention architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class ModalityEncoder(nn.Module):
    """EfficientNet-B0 backbone encoder for RGB or Thermal input."""

    def __init__(self, in_channels: int = 3, feature_dim: int = 256):
        super().__init__()
        backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        if in_channels == 1:
            # Adapt first conv for 1-channel thermal by averaging RGB weights
            orig_weight = backbone.features[0][0].weight.data  # (32,3,3,3)
            new_weight = orig_weight.mean(dim=1, keepdim=True)   # (32,1,3,3)
            backbone.features[0][0] = nn.Conv2d(
                1, 32, kernel_size=3, stride=2, padding=1, bias=False
            )
            backbone.features[0][0].weight.data = new_weight

        # Use features up to the last block (before classifier)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Get backbone output channels (1280 for EfficientNet-B0)
        backbone_out = 1280
        self.proj = nn.Sequential(
            nn.Conv2d(backbone_out, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, feature_dim, H/32, W/32)"""
        feat = self.features(x)
        return self.proj(feat)


class CrossModalAttention(nn.Module):
    """
    Bidirectional cross-modal attention between RGB and Thermal features.
    RGB queries attend to Thermal, and Thermal queries attend to RGB.
    """

    def __init__(self, feature_dim: int = 256, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim

        # RGB attends to Thermal
        self.attn_rgb_to_thm = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=0.1, batch_first=True
        )
        # Thermal attends to RGB
        self.attn_thm_to_rgb = nn.MultiheadAttention(
            feature_dim, num_heads, dropout=0.1, batch_first=True
        )

        self.norm_rgb = nn.LayerNorm(feature_dim)
        self.norm_thm = nn.LayerNorm(feature_dim)

        self.ffn_rgb = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.ffn_thm = nn.Sequential(
            nn.Linear(feature_dim, feature_dim * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim * 4, feature_dim),
        )
        self.norm_rgb2 = nn.LayerNorm(feature_dim)
        self.norm_thm2 = nn.LayerNorm(feature_dim)

    def forward(
        self, rgb_feat: torch.Tensor, thm_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            rgb_feat: (B, C, H, W)
            thm_feat: (B, C, H, W)
        Returns:
            rgb_out, thm_out: (B, C, H, W) each
        """
        B, C, H, W = rgb_feat.shape

        # Flatten spatial dims to sequence: (B, H*W, C)
        rgb_seq = rgb_feat.flatten(2).permute(0, 2, 1)
        thm_seq = thm_feat.flatten(2).permute(0, 2, 1)

        # Cross-attention
        rgb_attn, _ = self.attn_rgb_to_thm(rgb_seq, thm_seq, thm_seq)
        thm_attn, _ = self.attn_thm_to_rgb(thm_seq, rgb_seq, rgb_seq)

        # Residual + LayerNorm
        rgb_seq = self.norm_rgb(rgb_seq + rgb_attn)
        thm_seq = self.norm_thm(thm_seq + thm_attn)

        # FFN
        rgb_seq = self.norm_rgb2(rgb_seq + self.ffn_rgb(rgb_seq))
        thm_seq = self.norm_thm2(thm_seq + self.ffn_thm(thm_seq))

        # Reshape back to spatial
        rgb_out = rgb_seq.permute(0, 2, 1).reshape(B, C, H, W)
        thm_out = thm_seq.permute(0, 2, 1).reshape(B, C, H, W)

        return rgb_out, thm_out


class FusionModule(nn.Module):
    """
    Stacks N CrossModalAttention layers then produces a single fused feature map
    with learned modality weighting.
    """

    def __init__(self, feature_dim: int = 256, num_layers: int = 3, num_heads: int = 8):
        super().__init__()
        self.layers = nn.ModuleList(
            [CrossModalAttention(feature_dim, num_heads) for _ in range(num_layers)]
        )

        # Learned modality weights: w_rgb, w_thm
        self.modality_weight = nn.Parameter(torch.zeros(2))

        # Residual FFN on fused output
        self.residual_ffn = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(feature_dim),
        )

    def forward(
        self, rgb_feat: torch.Tensor, thm_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns fused (B,C,H,W), rgb_final, thm_final"""
        for layer in self.layers:
            rgb_feat, thm_feat = layer(rgb_feat, thm_feat)

        # Learned weighted combination
        weights = torch.softmax(self.modality_weight, dim=0)
        fused = weights[0] * rgb_feat + weights[1] * thm_feat

        # Residual FFN
        fused = fused + self.residual_ffn(fused)

        return fused, rgb_feat, thm_feat


class AnomalyFilter(nn.Module):
    """
    Cross-modal consistency filter:
    - Global score: is there a human (heat + body shape aligned)?
    - Local map: gates fused features spatially
    Hot debris / fire = low consistency = gated down
    """

    def __init__(self, feature_dim: int = 256):
        super().__init__()
        # Global consistency score
        self.global_mlp = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid(),
        )

        # Local consistency map
        self.local_conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(
        self, fused: torch.Tensor, rgb_feat: torch.Tensor, thm_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            filtered: fused features gated by local consistency
            global_score: (B,1) overall consistency
            local_map: (B,1,H,W) spatial consistency map
        """
        # Global pooling
        rgb_pooled = rgb_feat.flatten(2).mean(-1)   # (B, C)
        thm_pooled = thm_feat.flatten(2).mean(-1)   # (B, C)
        global_concat = torch.cat([rgb_pooled, thm_pooled], dim=1)
        global_score = self.global_mlp(global_concat)  # (B,1)

        # Local map
        local_concat = torch.cat([rgb_feat, thm_feat], dim=1)  # (B,2C,H,W)
        local_map = self.local_conv(local_concat)               # (B,1,H,W)

        # Gate fused features
        filtered = fused * local_map

        return filtered, global_score, local_map


class DetectionHead(nn.Module):
    """
    YOLO-style detection head.
    Outputs: bounding boxes + objectness + class scores
    """

    def __init__(
        self,
        feature_dim: int = 256,
        num_anchors: int = 3,
        num_classes: int = 1,
    ):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        out_channels = num_anchors * (5 + num_classes)  # cx,cy,w,h,obj + classes

        self.conv = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
            nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim // 2),
            nn.GELU(),
            nn.Conv2d(feature_dim // 2, out_channels, kernel_size=1),
        )


    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            boxes: (B, H, W, num_anchors, 5+num_classes)
            confidence: (B,) max objectness * max class score
        """
        B, C, H, W = x.shape

        raw = self.conv(x)  # (B, A*(5+C), H, W)
        raw = raw.permute(0, 2, 3, 1)  # (B, H, W, A*(5+C))
        raw = raw.reshape(B, H, W, self.num_anchors, 5 + self.num_classes)

        # Apply activations
        raw[..., :2] = torch.sigmoid(raw[..., :2])   # cx, cy
        raw[..., 2:4] = torch.exp(raw[..., 2:4])      # w, h (positive)
        raw[..., 4] = torch.sigmoid(raw[..., 4])       # objectness
        raw[..., 5:] = torch.sigmoid(raw[..., 5:])    # class scores

        # Overall confidence
        objectness = raw[..., 4]           # (B, H, W, A)
        class_score = raw[..., 5:].max(-1).values
        confidence = (objectness * class_score).reshape(B, -1).max(-1).values  # (B,)

        return raw, confidence


class SARFusionModel(nn.Module):
    """
    Complete SAR Fusion Model.
    Input: rgb (B,3,H,W), thermal (B,1,H,W) both in [0,1]
    """

    # Anchors for 320x320 input (relative to 10x10 grid)
    ANCHORS = [
        (0.28, 0.22), (0.38, 0.48), (0.9, 0.78),
    ]

    def __init__(
        self,
        feature_dim: int = 256,
        fusion_layers: int = 3,
        num_heads: int = 8,
        num_classes: int = 1,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        self.rgb_encoder = ModalityEncoder(in_channels=3, feature_dim=feature_dim)
        self.thm_encoder = ModalityEncoder(in_channels=1, feature_dim=feature_dim)
        self.fusion = FusionModule(feature_dim, fusion_layers, num_heads)
        self.anomaly_filter = AnomalyFilter(feature_dim)
        self.detection_head = DetectionHead(feature_dim, num_anchors=3, num_classes=num_classes)

    def forward(self, rgb: torch.Tensor, thermal: torch.Tensor) -> dict:
        # Encode both modalities
        rgb_feat = self.rgb_encoder(rgb)     # (B, C, H/32, W/32)
        thm_feat = self.thm_encoder(thermal) # (B, C, H/32, W/32)

        # Cross-modal fusion
        fused, rgb_final, thm_final = self.fusion(rgb_feat, thm_feat)

        # Anomaly filtering
        filtered, global_score, local_map = self.anomaly_filter(fused, rgb_final, thm_final)

        # Detection
        boxes, confidence = self.detection_head(filtered)

        # Modality weights for reporting
        weights = torch.softmax(self.fusion.modality_weight, dim=0).detach()

        return {
            "boxes": boxes,                     # (B, H, W, A, 5+C)
            "confidence": confidence,            # (B,)
            "consistency_score": global_score,   # (B,1)
            "local_consistency": local_map,      # (B,1,H,W)
            "rgb_features": rgb_final,
            "thm_features": thm_final,
            "modality_weights": weights,         # [w_rgb, w_thm]
        }

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(mode: str = "full") -> SARFusionModel:
    """Factory function for full or edge model."""
    if mode == "edge":
        return SARFusionModel(feature_dim=128, fusion_layers=2, num_heads=4)
    return SARFusionModel(feature_dim=256, fusion_layers=3, num_heads=8)


if __name__ == "__main__":
    model = build_model("full")
    print(f"Model parameters: {model.num_parameters:,}")

    # Dummy forward pass
    rgb = torch.randn(2, 3, 320, 320)
    thm = torch.randn(2, 1, 320, 320)
    out = model(rgb, thm)

    for k, v in out.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {tuple(v.shape)}")
        else:
            print(f"  {k}: {v}")
