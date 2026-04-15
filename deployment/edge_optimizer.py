"""
Edge Deployment Optimizer
Converts trained SAR model for real-time UAV inference via:
    - Post-training quantization (INT8 / FP16)
    - ONNX export
    - TorchScript tracing
    - Benchmark utilities
"""

import torch
import torch.nn as nn
from pathlib import Path
import time
import logging
from typing import Optional, Tuple

log = logging.getLogger("SAR-Edge")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ─────────────────────────────────────────────
# 1. EDGE-OPTIMIZED WRAPPER
# ─────────────────────────────────────────────

class EdgeSARModel(nn.Module):
    """
    Slim wrapper around SARFusionModel for edge deployment.
    Fuses batch norm, prunes unused heads, and provides a
    fixed-shape forward pass suitable for ONNX export.
    """

    def __init__(self, base_model: nn.Module, img_size: int = 320):
        super().__init__()
        self.model    = base_model
        self.img_size = img_size

    def forward(
        self,
        rgb:     torch.Tensor,    # (1, 3, H, W)
        thermal: torch.Tensor,    # (1, 1, H, W)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            confidence   : (1,)   — max detection confidence
            consistency  : (1,)   — cross-modal alignment score
            survival     : (1,)   — survival likelihood
        """
        out = self.model(rgb, thermal)
        conf        = out["confidence"].flatten().max().unsqueeze(0)
        consistency = out["consistency_score"]
        survival    = out["survival"].squeeze()
        return conf, consistency, survival


# ─────────────────────────────────────────────
# 2. QUANTIZATION
# ─────────────────────────────────────────────

def quantize_model(
    model: nn.Module,
    calibration_data: Optional[list] = None,
    mode: str = "dynamic",
) -> nn.Module:
    """
    mode = 'dynamic' : fast, no calibration data needed (recommended for edge)
    mode = 'static'  : better accuracy, requires calibration samples
    """
    model.eval()

    if mode == "dynamic":
        quantized = torch.quantization.quantize_dynamic(
            model,
            {nn.Linear, nn.Conv2d},
            dtype=torch.qint8,
        )
        log.info("Dynamic INT8 quantization applied")
        return quantized

    elif mode == "static":
        model.qconfig = torch.quantization.get_default_qconfig("fbgemm")
        torch.quantization.prepare(model, inplace=True)

        if calibration_data:
            with torch.no_grad():
                for rgb, thermal in calibration_data:
                    model(rgb, thermal)

        torch.quantization.convert(model, inplace=True)
        log.info("Static INT8 quantization applied")
        return model

    raise ValueError(f"Unknown quantization mode: {mode}")


# ─────────────────────────────────────────────
# 3. ONNX EXPORT
# ─────────────────────────────────────────────

def export_onnx(
    model:       nn.Module,
    output_path: str,
    img_size:    int = 320,
    opset:       int = 17,
) -> str:
    """Export EdgeSARModel to ONNX for TensorRT / ONNX Runtime deployment."""
    model.eval()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_rgb     = torch.randn(1, 3, img_size, img_size)
    dummy_thermal = torch.randn(1, 1, img_size, img_size)

    torch.onnx.export(
        model,
        (dummy_rgb, dummy_thermal),
        str(output_path),
        opset_version=opset,
        input_names=["rgb", "thermal"],
        output_names=["confidence", "consistency", "survival"],
        dynamic_axes={
            "rgb":         {0: "batch"},
            "thermal":     {0: "batch"},
            "confidence":  {0: "batch"},
            "consistency": {0: "batch"},
            "survival":    {0: "batch"},
        },
        do_constant_folding=True,
        verbose=False,
    )

    size_mb = output_path.stat().st_size / 1e6
    log.info(f"ONNX export → {output_path} ({size_mb:.1f} MB)")
    return str(output_path)


# ─────────────────────────────────────────────
# 4. TORCHSCRIPT TRACE
# ─────────────────────────────────────────────

def export_torchscript(
    model:       nn.Module,
    output_path: str,
    img_size:    int = 320,
) -> str:
    """Export to TorchScript for C++ / mobile / embedded deployment."""
    model.eval()
    output_path = Path(output_path)

    dummy_rgb     = torch.randn(1, 3, img_size, img_size)
    dummy_thermal = torch.randn(1, 1, img_size, img_size)

    with torch.no_grad():
        traced = torch.jit.trace(model, (dummy_rgb, dummy_thermal))

    traced.save(str(output_path))
    size_mb = output_path.stat().st_size / 1e6
    log.info(f"TorchScript export → {output_path} ({size_mb:.1f} MB)")
    return str(output_path)


# ─────────────────────────────────────────────
# 5. BENCHMARK UTILITY
# ─────────────────────────────────────────────

def benchmark(
    model:       nn.Module,
    img_size:    int   = 320,
    n_runs:      int   = 100,
    warmup:      int   = 10,
    device:      str   = "cpu",
) -> dict:
    """
    Measures inference latency and throughput.
    Run on the target hardware (Jetson / RPi) for real-world numbers.
    """
    model = model.to(device).eval()
    rgb     = torch.randn(1, 3, img_size, img_size, device=device)
    thermal = torch.randn(1, 1, img_size, img_size, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(rgb, thermal)

    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(rgb, thermal)
            if device == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)   # ms

    latencies  = sorted(latencies)
    avg_ms     = sum(latencies) / len(latencies)
    p95_ms     = latencies[int(0.95 * len(latencies))]
    throughput = 1000.0 / avg_ms

    results = {
        "device":         device,
        "img_size":       img_size,
        "avg_latency_ms": round(avg_ms, 2),
        "p95_latency_ms": round(p95_ms, 2),
        "fps":            round(throughput, 1),
        "real_time":      throughput >= 10.0,   # ≥10 FPS = real-time for SAR
    }

    log.info(
        f"Benchmark [{device}] | Avg: {avg_ms:.1f}ms | P95: {p95_ms:.1f}ms | "
        f"FPS: {throughput:.1f} | Real-time: {results['real_time']}"
    )
    return results


# ─────────────────────────────────────────────
# 6. FULL OPTIMIZATION PIPELINE
# ─────────────────────────────────────────────

def optimize_for_edge(
    checkpoint_path: str,
    output_dir:      str = "deploy/",
    img_size:        int = 320,
) -> dict:
    """
    Full pipeline:
        1. Load trained model
        2. Apply quantization
        3. Export ONNX + TorchScript
        4. Benchmark
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.fusion_model import build_sar_model

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    log.info(f"Loading checkpoint: {checkpoint_path}")
    ckpt  = torch.load(checkpoint_path, map_location="cpu")
    model = build_sar_model("edge")
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    edge_model = EdgeSARModel(model, img_size)

    # Quantize
    quant_model = quantize_model(edge_model, mode="dynamic")

    # Export
    onnx_path   = export_onnx(edge_model,   output_dir / "sar_model.onnx",     img_size)
    ts_path     = export_torchscript(model, output_dir / "sar_model_ts.pt",    img_size)
    qts_path    = export_torchscript(quant_model, output_dir / "sar_model_int8.pt", img_size)

    # Benchmark all variants
    results = {
        "fp32":  benchmark(edge_model,  img_size, device="cpu"),
        "int8":  benchmark(quant_model, img_size, device="cpu"),
    }

    if torch.cuda.is_available():
        results["cuda_fp16"] = benchmark(
            edge_model.half().cuda(), img_size, device="cuda"
        )

    # Summary
    summary = {
        "exports": {"onnx": onnx_path, "torchscript": ts_path, "int8": qts_path},
        "benchmarks": results,
        "recommended": "int8" if results["int8"]["real_time"] else "fp32",
    }

    (output_dir / "optimization_report.json").write_text(
        __import__("json").dumps(summary, indent=2)
    )
    log.info("Edge optimization complete.")
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAR Edge Optimization")
    parser.add_argument("--checkpoint", required=True, help="Path to trained .pt checkpoint")
    parser.add_argument("--output",     default="deploy/", help="Output directory")
    parser.add_argument("--img-size",   type=int, default=320)
    args = parser.parse_args()

    summary = optimize_for_edge(args.checkpoint, args.output, args.img_size)
    print("\n=== Optimization Summary ===")
    for mode, bench in summary["benchmarks"].items():
        print(f"  {mode:12s} | {bench['avg_latency_ms']:6.1f}ms | {bench['fps']:5.1f} FPS")
    print(f"\n  Recommended: {summary['recommended']}")