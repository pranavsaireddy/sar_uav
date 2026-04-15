"""
UAV Simulation Workflow
Simulates the complete SAR UAV pipeline:
    1. Synchronized RGB + Thermal frame capture
    2. Pre-processing and alignment
    3. On-device inference
    4. GPS-tagged output
    5. Mission reporting
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger("UAV-SIM")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ─────────────────────────────────────────────
# 1. UAV PLATFORM SIMULATION
# ─────────────────────────────────────────────

@dataclass
class GPSPosition:
    lat:      float = 17.385044       # Default: Hyderabad, India
    lon:      float = 78.486671
    altitude: float = 50.0            # metres AGL
    heading:  float = 0.0             # degrees (0=North)
    speed:    float = 5.0             # m/s

    def move(self, dt: float = 0.1) -> "GPSPosition":
        """Advance UAV position along current heading."""
        heading_rad = math.radians(self.heading)
        d_lat = (self.speed * dt * math.cos(heading_rad)) / 111_320
        d_lon = (self.speed * dt * math.sin(heading_rad)) / (111_320 * math.cos(math.radians(self.lat)))
        return GPSPosition(
            lat      = self.lat + d_lat,
            lon      = self.lon + d_lon,
            altitude = self.altitude,
            heading  = self.heading,
            speed    = self.speed,
        )


@dataclass
class IMUData:
    roll:  float = 0.0
    pitch: float = 0.0
    yaw:   float = 0.0
    ax:    float = 0.0   # m/s²
    ay:    float = 0.0
    az:    float = -9.81


@dataclass
class UAVTelemetry:
    gps:     GPSPosition
    imu:     IMUData
    battery: float           # 0.0 – 1.0
    timestamp: float = field(default_factory=time.time)


# ─────────────────────────────────────────────
# 2. SYNTHETIC FRAME GENERATOR
# ─────────────────────────────────────────────

class SyntheticFrameGenerator:
    """
    Generates synthetic RGB + Thermal frame pairs simulating
    disaster environments with human survivor signatures.
    """

    IMG_SIZE = 320

    def __init__(self, num_survivors: int = 3):
        self.survivors = [
            {
                "cx": random.uniform(0.2, 0.8),
                "cy": random.uniform(0.2, 0.8),
                "visible": True,
            }
            for _ in range(num_survivors)
        ]

    def _add_survivor_rgb(self, arr: np.ndarray, cx: float, cy: float):
        """Add a subtle human blob to RGB frame (partially occluded)."""
        H, W = arr.shape[:2]
        x0 = int((cx - 0.04) * W)
        y0 = int((cy - 0.08) * H)
        x1 = int((cx + 0.04) * W)
        y1 = int((cy + 0.08) * H)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(W, x1), min(H, y1)
        # Darken the region (human in debris)
        arr[y0:y1, x0:x1] = (arr[y0:y1, x0:x1] * 0.4 + np.array([80, 60, 50])).clip(0, 255)

    def _add_survivor_thermal(self, arr: np.ndarray, cx: float, cy: float):
        """Add a Gaussian heat signature to thermal frame."""
        H, W = arr.shape[:2]
        xs = np.linspace(0, 1, W)
        ys = np.linspace(0, 1, H)
        xx, yy = np.meshgrid(xs, ys)
        sigma_x, sigma_y = 0.04, 0.07
        heat = np.exp(-((xx - cx) ** 2) / (2 * sigma_x ** 2) - ((yy - cy) ** 2) / (2 * sigma_y ** 2))
        arr[:, :] = np.clip(arr[:, :] + (heat * 140).astype(np.uint8), 0, 255)

    def _add_false_positive(self, thermal: np.ndarray):
        """Add thermal false positives (hot debris, fire)."""
        for _ in range(random.randint(0, 3)):
            cx = random.random()
            cy = random.random()
            # Larger, hotter blob — no matching RGB structure
            H, W = thermal.shape
            xs = np.linspace(0, 1, W)
            ys = np.linspace(0, 1, H)
            xx, yy = np.meshgrid(xs, ys)
            sigma = random.uniform(0.03, 0.08)
            heat = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
            thermal[:, :] = np.clip(thermal + (heat * 200).astype(np.uint8), 0, 255)

    def generate(self, smoke_level: float = 0.3) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Returns:
            rgb      : (H, W, 3) uint8 — debris field with optional survivors
            thermal  : (H, W)    uint8 — heat map
            gt_boxes : list of {cx, cy, visible}
        """
        H = W = self.IMG_SIZE

        # Base RGB: rubble texture
        rgb = np.random.randint(60, 160, (H, W, 3), dtype=np.uint8)
        rgb += np.random.randint(-15, 15, (H, W, 3)).clip(-80, 80).astype(np.int16)
        rgb = rgb.clip(0, 255).astype(np.uint8)

        # Smoke overlay
        smoke = np.random.uniform(0, smoke_level, (H, W, 1))
        rgb = (rgb * (1 - smoke) + 180 * smoke).clip(0, 255).astype(np.uint8)

        # Base thermal: ambient temperature noise (cool ground)
        thermal = np.random.randint(20, 60, (H, W), dtype=np.uint8)

        # False positives first
        self._add_false_positive(thermal)

        # Survivors
        gt_boxes = []
        for s in self.survivors:
            if s["visible"]:
                self._add_survivor_rgb(rgb, s["cx"], s["cy"])
                self._add_survivor_thermal(thermal, s["cx"], s["cy"])
            gt_boxes.append({"cx": s["cx"], "cy": s["cy"], "visible": s["visible"]})

        return rgb, thermal, gt_boxes


# ─────────────────────────────────────────────
# 3. FRAME PREPROCESSOR
# ─────────────────────────────────────────────

def preprocess_frames(
    rgb:     np.ndarray,
    thermal: np.ndarray,
    device:  torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize and batch numpy frames → model-ready tensors."""
    # RGB: HWC uint8 → CHW float [0,1]
    r = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    r = r.unsqueeze(0).to(device)

    # Thermal: HW uint8 → 1HW float [0,1]
    t = torch.from_numpy(thermal).float() / 255.0
    t = t.unsqueeze(0).unsqueeze(0).to(device)

    return r, t


# ─────────────────────────────────────────────
# 4. MISSION PLANNER
# ─────────────────────────────────────────────

@dataclass
class MissionConfig:
    name:           str   = "SAR-MISSION-001"
    search_area_km: float = 0.5            # km × km grid
    altitude_m:     float = 50.0
    speed_ms:       float = 5.0
    fps:            float = 10.0           # inference rate
    max_duration_s: float = 300.0          # 5 minutes
    conf_threshold: float = 0.45
    start_lat:      float = 17.385044
    start_lon:      float = 78.486671
    smoke_level:    float = 0.4            # 0=clear, 1=heavy smoke


@dataclass
class DetectionEvent:
    frame_id:   str
    timestamp:  float
    gps:        GPSPosition
    confidence: float
    consistency: float
    survival:   float
    boxes:      List[Dict]
    explanation: str


@dataclass
class MissionReport:
    mission:     MissionConfig
    start_time:  float
    end_time:    float
    frames_processed: int
    detections:  List[DetectionEvent]
    false_positives_suppressed: int
    avg_latency_ms: float
    coverage_pct:   float


# ─────────────────────────────────────────────
# 5. UAV SIMULATION RUNNER
# ─────────────────────────────────────────────

class UAVSimulation:
    """
    Full end-to-end SAR mission simulation.
    Uses the trained model (or random-weight demo) to process
    synthetic frames and log detection events.
    """

    def __init__(
        self,
        cfg: MissionConfig,
        checkpoint: Optional[str] = None,
    ):
        self.cfg     = cfg
        self.device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model   = self._load_model(checkpoint)
        self.gen     = SyntheticFrameGenerator(num_survivors=3)
        self.gps     = GPSPosition(
            lat=cfg.start_lat, lon=cfg.start_lon,
            altitude=cfg.altitude_m, speed=cfg.speed_ms,
        )
        log.info(f"Simulation '{cfg.name}' ready on {self.device}")

    def _load_model(self, checkpoint: Optional[str]):
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from models.fusion_model import build_sar_model

        model = build_sar_model("edge").to(self.device).eval()
        if checkpoint and Path(checkpoint).exists():
            ckpt = torch.load(checkpoint, map_location=self.device)
            model.load_state_dict(ckpt["model_state"])
            log.info(f"Loaded model from {checkpoint}")
        else:
            log.warning("Demo mode: random weights — detection values are illustrative")
        return model

    def _lawnmower_heading(self, frame_idx: int) -> float:
        """Classic SAR lawnmower pattern: alternating E-W strips."""
        strip = frame_idx // 40
        return 90.0 if strip % 2 == 0 else 270.0

    @torch.no_grad()
    def _infer_frame(
        self,
        rgb_arr: np.ndarray,
        thermal_arr: np.ndarray,
    ) -> Dict:
        t0  = time.perf_counter()
        rgb, thm = preprocess_frames(rgb_arr, thermal_arr, self.device)
        out = self.model(rgb, thm)

        conf        = float(out["confidence"].max())
        consistency = float(out["consistency_score"])
        survival    = float(out["survival"].squeeze())
        latency_ms  = (time.perf_counter() - t0) * 1000

        return {
            "confidence":   round(conf, 4),
            "consistency":  round(consistency, 4),
            "survival":     round(survival, 4),
            "latency_ms":   round(latency_ms, 2),
        }

    async def _frame_loop(self) -> AsyncGenerator[Dict, None]:
        """Async generator: yields one processed frame per tick."""
        frame_idx = 0
        dt = 1.0 / self.cfg.fps

        while True:
            # Move UAV
            self.gps.heading = self._lawnmower_heading(frame_idx)
            self.gps = self.gps.move(dt)

            # Generate synthetic frames
            rgb_arr, thm_arr, gt = self.gen.generate(self.cfg.smoke_level)

            # Infer
            result = self._infer_frame(rgb_arr, thm_arr)
            result["frame_id"]  = f"F{frame_idx:06d}"
            result["timestamp"] = time.time()
            result["gps"]       = asdict(self.gps)
            result["gt_boxes"]  = gt

            yield result
            frame_idx += 1
            await asyncio.sleep(dt)

    async def run(self) -> MissionReport:
        """Execute full mission and collect results."""
        start = time.time()
        frames_total     = int(self.cfg.max_duration_s * self.cfg.fps)
        detections:  List[DetectionEvent] = []
        fp_suppressed    = 0
        total_latency    = 0.0
        frames_processed = 0

        log.info(f"Mission start | Frames: {frames_total} | FPS: {self.cfg.fps}")

        async for frame in self._frame_loop():
            frames_processed += 1
            total_latency    += frame["latency_ms"]
            conf        = frame["confidence"]
            consistency = frame["consistency"]

            detected    = conf > self.cfg.conf_threshold and consistency > 0.2

            if conf > self.cfg.conf_threshold and not detected:
                fp_suppressed += 1
                log.debug(f"[{frame['frame_id']}] FP suppressed | conf={conf:.2f} consistency={consistency:.2f}")

            if detected:
                event = DetectionEvent(
                    frame_id    = frame["frame_id"],
                    timestamp   = frame["timestamp"],
                    gps         = GPSPosition(**frame["gps"]),
                    confidence  = conf,
                    consistency = consistency,
                    survival    = frame["survival"],
                    boxes       = frame["gt_boxes"],
                    explanation = _make_explanation(conf, consistency),
                )
                detections.append(event)
                log.info(
                    f"DETECTION [{frame['frame_id']}] | "
                    f"Conf: {conf:.3f} | Cons: {consistency:.3f} | "
                    f"GPS: ({event.gps.lat:.5f}, {event.gps.lon:.5f})"
                )

            # Print live status every 50 frames
            if frames_processed % 50 == 0:
                pct = frames_processed / frames_total * 100
                log.info(
                    f"Progress: {pct:.0f}% | Frames: {frames_processed}/{frames_total} | "
                    f"Detections: {len(detections)} | FP suppressed: {fp_suppressed} | "
                    f"Avg lat: {total_latency/frames_processed:.1f}ms"
                )

            if frames_processed >= frames_total:
                break

        report = MissionReport(
            mission              = self.cfg,
            start_time           = start,
            end_time             = time.time(),
            frames_processed     = frames_processed,
            detections           = detections,
            false_positives_suppressed = fp_suppressed,
            avg_latency_ms       = round(total_latency / max(frames_processed, 1), 2),
            coverage_pct         = min(100.0, frames_processed / frames_total * 100),
        )
        return report


def _make_explanation(conf: float, consistency: float) -> str:
    if conf > 0.8 and consistency > 0.7:
        return "High-confidence detection — strong RGB-thermal alignment."
    if conf > 0.6:
        return "Moderate detection — recommend UAV hover and secondary scan."
    return "Low-confidence detection — flagged for review."


# ─────────────────────────────────────────────
# 6. REPORT PRINTER
# ─────────────────────────────────────────────

def print_report(report: MissionReport):
    print("\n" + "=" * 60)
    print(f"  SAR MISSION REPORT: {report.mission.name}")
    print("=" * 60)
    print(f"  Duration:          {report.end_time - report.start_time:.1f}s")
    print(f"  Frames processed:  {report.frames_processed}")
    print(f"  Coverage:          {report.coverage_pct:.1f}%")
    print(f"  Avg latency:       {report.avg_latency_ms:.1f}ms ({1000/max(report.avg_latency_ms,1):.1f} FPS)")
    print(f"  Total detections:  {len(report.detections)}")
    print(f"  FP suppressed:     {report.false_positives_suppressed}")
    print(f"  Smoke level:       {report.mission.smoke_level:.0%}")
    print("-" * 60)

    if report.detections:
        print(f"\n  SURVIVOR LOCATIONS ({len(report.detections)} detected):")
        for i, ev in enumerate(report.detections, 1):
            print(
                f"    [{i}] Frame {ev.frame_id} | "
                f"GPS: ({ev.gps.lat:.5f}°N, {ev.gps.lon:.5f}°E) | "
                f"Conf: {ev.confidence:.2f} | Survival: {ev.survival:.2f}"
            )
            print(f"         {ev.explanation}")
    else:
        print("\n  No survivors detected in this mission segment.")

    print("=" * 60 + "\n")

    # Save JSON report
    report_path = Path("mission_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "mission":     report.mission.name,
            "frames":      report.frames_processed,
            "detections":  len(report.detections),
            "fp_suppressed": report.false_positives_suppressed,
            "avg_latency_ms": report.avg_latency_ms,
            "survivor_gps": [
                {"lat": e.gps.lat, "lon": e.gps.lon, "conf": e.confidence}
                for e in report.detections
            ]
        }, f, indent=2)
    print(f"  Report saved → {report_path}")


# ─────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SAR UAV Simulation")
    parser.add_argument("--checkpoint",  default=None,         help="Model checkpoint path")
    parser.add_argument("--duration",    type=float, default=60.0, help="Mission duration (s)")
    parser.add_argument("--fps",         type=float, default=10.0)
    parser.add_argument("--smoke",       type=float, default=0.4,  help="Smoke level 0-1")
    parser.add_argument("--mission",     default="SAR-MISSION-001")
    args = parser.parse_args()

    cfg = MissionConfig(
        name           = args.mission,
        max_duration_s = args.duration,
        fps            = args.fps,
        smoke_level    = args.smoke,
    )

    sim    = UAVSimulation(cfg, checkpoint=args.checkpoint)
    report = asyncio.run(sim.run())
    print_report(report)