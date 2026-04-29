"""
SAR FastAPI Inference Server
Real-time multimodal detection API with:
    - REST endpoint for single-frame inference
    - WebSocket for live UAV feed streaming
    - GPS location tagging
    - Detection history logging
    - Health monitoring
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ─── Local imports ────────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from models.fusion_model import build_sar_model, SARFusionModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("SAR-API")


# ─────────────────────────────────────────────
# 1. PYDANTIC SCHEMAS
# ─────────────────────────────────────────────

class GPSCoord(BaseModel):
    lat:      float = Field(..., description="Latitude in decimal degrees")
    lon:      float = Field(..., description="Longitude in decimal degrees")
    altitude: float = Field(default=0.0, description="Altitude in meters")
    heading:  float = Field(default=0.0, description="UAV heading in degrees")


class InferenceRequest(BaseModel):
    rgb_b64:     str           = Field(..., description="Base64-encoded RGB JPEG/PNG")
    thermal_b64: str           = Field(..., description="Base64-encoded thermal PNG (grayscale)")
    gps:         Optional[GPSCoord] = None
    frame_id:    Optional[str] = None
    timestamp:   Optional[str] = None


class BoundingBox(BaseModel):
    cx:         float
    cy:         float
    w:          float
    h:          float
    confidence: float


class DetectionResult(BaseModel):
    detected:           bool
    confidence:         float
    consistency_score:  float
    survival_likelihood: float
    bounding_boxes:     List[BoundingBox]
    gps_location:       Optional[GPSCoord]
    explanation:        str
    frame_id:           str
    timestamp:          str
    inference_ms:       float


class SystemStatus(BaseModel):
    status:         str
    model_loaded:   bool
    device:         str
    total_frames:   int
    detections:     int
    avg_latency_ms: float
    uptime_seconds: float


# ─────────────────────────────────────────────
# 2. MODEL INFERENCE ENGINE
# ─────────────────────────────────────────────

class InferenceEngine:
    """Thread-safe inference engine wrapping the SAR fusion model."""

    IMG_SIZE = 320

    def __init__(self, checkpoint_path: Optional[str] = None, device: str = "auto"):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and device == "auto"
            else "cpu" if device == "auto" else device
        )
        self.model   = self._load_model(checkpoint_path)
        self.lock    = asyncio.Lock()
        self.history: deque = deque(maxlen=1000)
        self.frame_count     = 0
        self.detect_count    = 0
        self.total_latency   = 0.0
        self.start_time      = time.time()
        log.info(f"Inference engine ready on {self.device}")

    def _load_model(self, checkpoint_path: Optional[str]) -> SARFusionModel:
        model = build_sar_model("edge")

        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            model.load_state_dict(ckpt["model_state"])
            log.info(f"Loaded checkpoint: {checkpoint_path}")
        else:
            log.warning("No checkpoint — running with random weights (demo mode)")

        model = model.to(self.device).eval()
        return model

    def _decode_image(self, b64: str, channels: int) -> torch.Tensor:
        """Decode base64 image → normalized tensor."""
        from PIL import Image

        raw  = base64.b64decode(b64)
        img  = Image.open(io.BytesIO(raw))
        mode = "RGB" if channels == 3 else "L"
        img  = img.convert(mode).resize((self.IMG_SIZE, self.IMG_SIZE))
        arr  = np.array(img, dtype=np.float32) / 255.0

        if channels == 3:
            t = torch.from_numpy(arr).permute(2, 0, 1)          # (3, H, W)
        else:
            t = torch.from_numpy(arr).unsqueeze(0)               # (1, H, W)

        return t.unsqueeze(0).to(self.device)                    # (1, C, H, W)

    def _parse_boxes(
        self, raw_boxes: torch.Tensor, confidence: torch.Tensor, threshold: float = 0.4
    ) -> List[BoundingBox]:
        boxes = []
        H, W, A = raw_boxes.shape[:3]
        for gy in range(H):
            for gx in range(W):
                for a in range(A):
                    conf = confidence[0, gy, gx, a, 0].item()
                    if conf > threshold:
                        cx, cy, w, h = raw_boxes[0, gy, gx, a, :4].sigmoid().tolist()
                        boxes.append(BoundingBox(cx=cx, cy=cy, w=w, h=h, confidence=round(conf, 4)))
        return boxes

    @torch.no_grad()
    async def infer(self, req: InferenceRequest) -> DetectionResult:
        async with self.lock:
            t0 = time.perf_counter()

            rgb     = self._decode_image(req.rgb_b64,     channels=3)
            thermal = self._decode_image(req.thermal_b64, channels=1)

            outputs = self.model(rgb, thermal)

            conf        = float(outputs["confidence"].max())
            consistency = float(outputs["consistency_score"])
            survival    = float(outputs["survival"].squeeze())

            boxes = self._parse_boxes(
                outputs["boxes"].cpu(), outputs["confidence"].cpu()
            )

            detected = conf > 0.5 and consistency > 0.25

            latency_ms = (time.perf_counter() - t0) * 1000
            self.total_latency += latency_ms
            self.frame_count   += 1
            if detected:
                self.detect_count += 1

            fid = req.frame_id or str(uuid.uuid4())[:8]
            ts  = req.timestamp or datetime.now(timezone.utc).isoformat()

            result = DetectionResult(
                detected            = detected,
                confidence          = round(conf, 4),
                consistency_score   = round(consistency, 4),
                survival_likelihood = round(survival, 4),
                bounding_boxes      = boxes,
                gps_location        = req.gps,
                explanation         = _explain(conf, consistency, detected),
                frame_id            = fid,
                timestamp           = ts,
                inference_ms        = round(latency_ms, 2),
            )
            self.history.append(result.model_dump())
            return result

    def get_status(self) -> SystemStatus:
        avg = self.total_latency / max(self.frame_count, 1)
        return SystemStatus(
            status         = "operational",
            model_loaded   = True,
            device         = str(self.device),
            total_frames   = self.frame_count,
            detections     = self.detect_count,
            avg_latency_ms = round(avg, 2),
            uptime_seconds = round(time.time() - self.start_time, 1),
        )


def _explain(conf: float, consistency: float, detected: bool) -> str:
    if not detected:
        if conf > 0.5 and consistency < 0.25:
            return "Thermal hotspot detected but no matching RGB body shape — likely false positive suppressed."
        return "No confident human detection in current frame."
    if conf > 0.8 and consistency > 0.7:
        return "Detected human — strong alignment between RGB body structure and thermal heat signature."
    return "Probable human detection — moderate RGB-thermal alignment. Recommend follow-up scan."


# ─────────────────────────────────────────────
# 3. FASTAPI APPLICATION
# ─────────────────────────────────────────────

app = FastAPI(
    title="SAR Multimodal Detection API",
    description="UAV-based RGB-Thermal Human Detection for Search and Rescue",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global engine (initialized on startup)
engine: Optional[InferenceEngine] = None
detection_log: List[Dict] = []


@app.on_event("startup")
async def startup():
    global engine
    checkpoint = Path("checkpoints/best_sar_model.pt")
    engine = InferenceEngine(
        checkpoint_path=str(checkpoint) if checkpoint.exists() else None
    )
    log.info("SAR API server started")


# ─────────────────────────────────────────────
# 4. REST ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/status", response_model=SystemStatus)
async def status():
    if engine is None:
        raise HTTPException(503, "Engine not initialized")
    return engine.get_status()


@app.post("/detect", response_model=DetectionResult)
async def detect(req: InferenceRequest, background_tasks: BackgroundTasks):
    """
    Submit synchronized RGB + thermal frames for human detection.

    Returns detection result with bounding boxes, GPS location,
    confidence, and cross-modal consistency score.
    """
    if engine is None:
        raise HTTPException(503, "Inference engine not ready")

    try:
        result = await engine.infer(req)
        background_tasks.add_task(_log_detection, result)
        return result
    except Exception as e:
        log.error(f"Inference error: {e}")
        raise HTTPException(500, f"Inference failed: {str(e)}")


@app.get("/history")
async def history(limit: int = 50):
    """Returns recent detection history."""
    if engine is None:
        raise HTTPException(503, "Engine not ready")
    recent = list(engine.history)[-limit:]
    return {"count": len(recent), "results": recent}


@app.get("/detections/summary")
async def detection_summary():
    """Aggregated detection statistics for dashboard."""
    if engine is None:
        raise HTTPException(503, "Engine not ready")

    hist   = list(engine.history)
    total  = len(hist)
    dets   = sum(1 for h in hist if h["detected"])
    gps_pts = [
        {"lat": h["gps_location"]["lat"], "lon": h["gps_location"]["lon"],
         "conf": h["confidence"]}
        for h in hist if h.get("gps_location") and h["detected"]
    ]

    return {
        "total_frames":     total,
        "total_detections": dets,
        "detection_rate":   round(dets / max(total, 1), 4),
        "survivor_locations": gps_pts,
        "avg_confidence":   round(sum(h["confidence"] for h in hist) / max(total, 1), 4),
    }


# ─────────────────────────────────────────────
# 5. WEBSOCKET LIVE STREAMING
# ─────────────────────────────────────────────

class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        self.active.remove(ws)

    async def broadcast(self, data: dict):
        for ws in list(self.active):
            try:
                await ws.send_json(data)
            except Exception:
                self.disconnect(ws)


manager = ConnectionManager()


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    WebSocket endpoint for real-time UAV feed processing.
    Client sends: {"rgb_b64": "...", "thermal_b64": "...", "gps": {...}}
    Server streams: DetectionResult JSON
    """
    await manager.connect(websocket)
    log.info(f"WebSocket connected: {websocket.client}")

    try:
        while True:
            raw = await websocket.receive_text()
            payload = json.loads(raw)

            req    = InferenceRequest(**payload)
            result = await engine.infer(req)
            await websocket.send_json(result.model_dump())

    except WebSocketDisconnect:
        manager.disconnect(websocket)
        log.info("WebSocket disconnected")
    except Exception as e:
        log.error(f"WebSocket error: {e}")
        manager.disconnect(websocket)


@app.websocket("/ws/broadcast")
async def websocket_broadcast(websocket: WebSocket):
    """Subscribe to all detection events (monitor dashboard)."""
    await manager.connect(websocket)
    try:
        while True:
            await asyncio.sleep(0.1)   # keep alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ─────────────────────────────────────────────
# 6. UTILITIES
# ─────────────────────────────────────────────

async def _log_detection(result: DetectionResult):
    """Background task: persist detection to log file."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"detections_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(result.model_dump()) + "\n")


# ─────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,        # single worker — GPU not fork-safe
        log_level="info",
    )