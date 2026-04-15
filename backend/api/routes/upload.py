"""
POST /upload — accept RGB + thermal image files, run inference, return DetectionResult.
"""

from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

router = APIRouter(tags=["upload"])

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


@router.post("/upload")
async def upload_files(
    rgb_file: UploadFile = File(..., description="RGB image (JPEG or PNG)"),
    thermal_file: UploadFile = File(..., description="Thermal image (PNG or JPEG)"),
    lat: Optional[float] = Form(None),
    lon: Optional[float] = Form(None),
    altitude: Optional[float] = Form(0.0),
):
    """
    Upload a paired RGB + thermal image for inference.
    Returns a full DetectionResult JSON.
    """
    from backend.api.main import get_engine, get_history, get_stats

    # Validate content type
    allowed_types = {"image/jpeg", "image/png", "image/jpg", "image/webp"}
    for f in (rgb_file, thermal_file):
        if f.content_type and f.content_type not in allowed_types:
            raise HTTPException(422, f"Unsupported file type: {f.content_type}")

    rgb_bytes = await rgb_file.read()
    thermal_bytes = await thermal_file.read()

    if len(rgb_bytes) > MAX_FILE_SIZE or len(thermal_bytes) > MAX_FILE_SIZE:
        raise HTTPException(413, "File too large (max 10 MB per file)")

    gps = None
    if lat is not None and lon is not None:
        gps = {"lat": lat, "lon": lon, "altitude": altitude or 0.0}

    engine = get_engine()
    result = engine.infer(rgb_bytes, thermal_bytes, gps)
    result["timestamp"] = datetime.utcnow().isoformat() + "Z"

    # Store in history
    history = get_history()
    history.append(result)
    if len(history) > int(1000):
        history.pop(0)

    # Update stats
    stats = get_stats()
    stats["total_frames"] += 1
    if result["detected"]:
        stats["total_detections"] += 1
    else:
        stats["fp_suppressed"] += 1
    stats["inference_ms_sum"] += result["inference_ms"]
    stats["confidence_sum"] += result["confidence"]
    stats["consistency_sum"] += result["consistency_score"]

    return result
