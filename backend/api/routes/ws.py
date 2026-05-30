"""
WebSocket /ws/live — real-time frame streaming
WebSocket /ws/monitor — broadcast subscriber
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(tags=["websocket"])

# Global set of monitor subscribers
_monitor_clients: set[WebSocket] = set()
_inference_lock = asyncio.Lock()


async def _broadcast(message: dict):
    """Broadcast to all monitor subscribers."""
    dead = set()
    for ws in _monitor_clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.add(ws)
    _monitor_clients.difference_update(dead)


@router.websocket("/ws/live")
async def ws_live(websocket: WebSocket):
    """
    Client sends JSON frames, server responds with DetectionResult.
    Frame format: { rgb_b64: string, thermal_b64: string, gps?: {...} }
    """
    from backend.api.main import get_engine, get_history, get_stats

    await websocket.accept()
    engine = get_engine()

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)

            import base64
            rgb_bytes = base64.b64decode(data.get("rgb_b64", ""))
            thermal_bytes = base64.b64decode(data.get("thermal_b64", ""))
            gps = data.get("gps")

            async with _inference_lock:
                result = engine.infer(rgb_bytes, thermal_bytes, gps)

            result["timestamp"] = datetime.utcnow().isoformat() + "Z"

            # Store & update stats
            history = get_history()
            history.append(result)
            if len(history) > 1000:
                history.pop(0)

            s = get_stats()
            s["total_frames"] += 1
            if result["detected"]:
                s["total_detections"] += 1
            s["inference_ms_sum"] += result["inference_ms"]
            s["confidence_sum"] += result["confidence"]
            s["consistency_sum"] += result["consistency_score"]

            await websocket.send_json(result)
            await _broadcast(result)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass


@router.websocket("/ws/monitor")
async def ws_monitor(websocket: WebSocket):
    """Subscribe to all detection events broadcast."""
    await websocket.accept()
    _monitor_clients.add(websocket)
    try:
        while True:
            # Keep connection alive with ping
            await asyncio.sleep(15)
            await websocket.send_json({"type": "ping"})
    except WebSocketDisconnect:
        _monitor_clients.discard(websocket)
    except Exception:
        _monitor_clients.discard(websocket)
