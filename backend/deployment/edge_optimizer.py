"""
SAR UAV Detection System - Edge Optimizer
Quantize and export to ONNX for deployment.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from backend.models.fusion_model import build_model


def export_onnx(
    checkpoint_path: str,
    output_path: str,
    model_mode: str = "edge",
    img_size: int = 320,
    opset: int = 17,
):
    """Export model to ONNX."""
    device = torch.device("cpu")  # ONNX export on CPU
    model = build_model(model_mode).to(device)

    ckpt = torch.load(checkpoint_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state, strict=False)
    model.eval()

    dummy_rgb = torch.randn(1, 3, img_size, img_size)
    dummy_thm = torch.randn(1, 1, img_size, img_size)

    # Wrap for ONNX (single output)
    class OnnxWrapper(nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, rgb, thermal):
            out = self.m(rgb, thermal)
            return out["confidence"], out["survival_likelihood"], out["consistency_score"]

    wrapped = OnnxWrapper(model)

    import onnx
    torch.onnx.export(
        wrapped,
        (dummy_rgb, dummy_thm),
        output_path,
        input_names=["rgb", "thermal"],
        output_names=["confidence", "survival_likelihood", "consistency_score"],
        dynamic_axes={
            "rgb": {0: "batch"},
            "thermal": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    # Verify
    model_onnx = onnx.load(output_path)
    onnx.checker.check_model(model_onnx)
    size_mb = Path(output_path).stat().st_size / 1e6
    print(f"ONNX export complete: {output_path} ({size_mb:.1f} MB)")
    return output_path


def benchmark_onnx(onnx_path: str, img_size: int = 320, n_runs: int = 200):
    """Benchmark ONNX model on GPU."""
    import time
    import numpy as np

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed")
        return

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = ort.InferenceSession(onnx_path, providers=providers)

    rgb = np.random.rand(1, 3, img_size, img_size).astype(np.float32)
    thm = np.random.rand(1, 1, img_size, img_size).astype(np.float32)

    # Warmup
    for _ in range(10):
        sess.run(None, {"rgb": rgb, "thermal": thm})

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {"rgb": rgb, "thermal": thm})
        times.append((time.perf_counter() - t0) * 1000)

    avg_ms = sum(times) / len(times)
    fps = 1000 / avg_ms
    print(f"ONNX Benchmark ({n_runs} runs): {avg_ms:.2f} ms avg | {fps:.0f} FPS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="checkpoints/sar_edge.onnx")
    parser.add_argument("--mode", default="edge", choices=["full", "edge"])
    parser.add_argument("--img-size", type=int, default=320)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()

    onnx_path = export_onnx(args.checkpoint, args.output, args.mode, args.img_size)
    if args.benchmark:
        benchmark_onnx(onnx_path, args.img_size)
