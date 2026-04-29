"""
GET /stats, GET /history, DELETE /history, GET /detections/map
"""

import time
from typing import Optional

from fastapi import APIRouter, Query

router = APIRouter(tags=["stats"])


@router.get("/stats")
async def get_stats_endpoint():
    from backend.api.main import get_engine, get_stats, get_start_time

    engine = get_engine()
    s = get_stats()
    n = max(1, s["total_frames"])
    elapsed = time.time() - get_start_time()

    return {
        "total_frames": s["total_frames"],
        "total_detections": s["total_detections"],
        "false_positives_suppressed": s["fp_suppressed"],
        "avg_inference_ms": round(s["inference_ms_sum"] / n, 2),
        "avg_confidence": round(s["confidence_sum"] / n, 4),
        "avg_consistency": round(s["consistency_sum"] / n, 4),
        "model_loaded": engine.is_ready,
        "device": str(engine.device),
        "uptime_seconds": round(elapsed, 1),
        "detections_per_minute": round(s["total_detections"] / max(1, elapsed / 60), 2),
    }


@router.get("/history")
async def get_history_endpoint(limit: int = Query(50, ge=1, le=500)):
    from backend.api.main import get_history
    history = get_history()
    return history[-limit:][::-1]  # most recent first


@router.delete("/history")
async def clear_history():
    from backend.api.main import get_history
    history = get_history()
    n = len(history)
    history.clear()
    return {"cleared": n}


@router.get("/detections/map")
async def get_gps_points():
    from backend.api.main import get_history
    history = get_history()
    points = []
    for r in history:
        if r.get("detected") and r.get("gps_location"):
            gps = r["gps_location"]
            points.append({
                "lat": gps.get("lat", 0),
                "lon": gps.get("lon", 0),
                "altitude": gps.get("altitude", 0),
                "confidence": r["confidence"],
                "frame_id": r["frame_id"],
                "timestamp": r.get("timestamp", ""),
            })
    return points
