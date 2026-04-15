"""
SAR UAV Detection System - FastAPI Application
"""

from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

load_dotenv()

# Shared inference engine and detection history (module-level singletons)
from backend.inference.engine import InferenceEngine

_engine: InferenceEngine | None = None
_start_time: float = time.time()
_detection_history: list[dict] = []
_stats = {
    "total_frames": 0,
    "total_detections": 0,
    "fp_suppressed": 0,
    "inference_ms_sum": 0.0,
    "confidence_sum": 0.0,
    "consistency_sum": 0.0,
}


def get_engine() -> InferenceEngine:
    global _engine
    if _engine is None:
        raise RuntimeError("Engine not initialized")
    return _engine


def get_history() -> list[dict]:
    return _detection_history


def get_stats() -> dict:
    return _stats


def get_start_time() -> float:
    return _start_time


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    model_path = os.getenv("MODEL_PATH", "checkpoints/best_sar_model.pt")
    model_mode = os.getenv("MODEL_MODE", "full")
    img_size = int(os.getenv("IMG_SIZE", "320"))
    conf_threshold = float(os.getenv("CONF_THRESHOLD", "0.45"))
    device = os.getenv("DEVICE", "cuda")

    _engine = InferenceEngine(
        checkpoint_path=model_path,
        model_mode=model_mode,
        img_size=img_size,
        conf_threshold=conf_threshold,
        device=device,
    )
    print(f"SAR backend ready on device: {_engine.device}")
    yield
    print("SAR backend shutting down")


app = FastAPI(
    title="SAR UAV Detection API",
    description="Multimodal RGB-Thermal Fusion for Human Survivor Detection",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Max upload size: 10 MB per file
from fastapi import HTTPException
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Register routes
from backend.api.routes import detect, upload, stats, ws

app.include_router(detect.router)
app.include_router(upload.router)
app.include_router(stats.router)
app.include_router(ws.router)


@app.get("/health")
async def health():
    engine = get_engine()
    return {
        "status": "ok",
        "model_loaded": engine.is_ready,
        "device": str(engine.device),
        "uptime_seconds": round(time.time() - _start_time, 1),
    }


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
    )
