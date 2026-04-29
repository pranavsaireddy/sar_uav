"""
POST /detect — base64 inference endpoint
"""

import base64
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(tags=["detect"])


class InferenceRequest(BaseModel):
    rgb_b64: str
    thermal_b64: str
    gps: Optional[dict] = None


@router.post("/detect")
async def detect(req: InferenceRequest):
    from backend.api.main import get_engine, get_history, get_stats

    try:
        rgb_bytes = base64.b64decode(req.rgb_b64)
        thermal_bytes = base64.b64decode(req.thermal_b64)
    except Exception:
        raise HTTPException(422, "Invalid base64 data")

    engine = get_engine()
    result = engine.infer(rgb_bytes, thermal_bytes, req.gps)
    result["timestamp"] = datetime.utcnow().isoformat() + "Z"

    history = get_history()
    history.append(result)
    if len(history) > 1000:
        history.pop(0)

    stats = get_stats()
    stats["total_frames"] += 1
    if result["detected"]:
        stats["total_detections"] += 1
    stats["inference_ms_sum"] += result["inference_ms"]
    stats["confidence_sum"] += result["confidence"]
    stats["consistency_sum"] += result["consistency_score"]

    return result
