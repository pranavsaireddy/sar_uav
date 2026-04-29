"""
SAR UAV Detection System - Pydantic Schemas
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class BoundingBox(BaseModel):
    cx: float = Field(..., ge=0.0, le=1.0)
    cy: float = Field(..., ge=0.0, le=1.0)
    w: float = Field(..., ge=0.0, le=1.0)
    h: float = Field(..., ge=0.0, le=1.0)
    confidence: float = Field(..., ge=0.0, le=1.0)


class GPSLocation(BaseModel):
    lat: float
    lon: float
    altitude: float = 0.0


class ModalityWeights(BaseModel):
    rgb: float
    thermal: float


class DetectionResult(BaseModel):
    detected: bool
    confidence: float = Field(..., ge=0.0, le=1.0)
    consistency_score: float = Field(..., ge=0.0, le=1.0)
    survival_likelihood: float = Field(..., ge=0.0, le=1.0)
    bounding_boxes: list[BoundingBox] = []
    gps_location: Optional[GPSLocation] = None
    explanation: str
    frame_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    inference_ms: float
    modality_weights: Optional[ModalityWeights] = None


class InferenceRequest(BaseModel):
    """For base64 POST /detect endpoint."""
    rgb_b64: str
    thermal_b64: str
    gps: Optional[GPSLocation] = None


class SystemStats(BaseModel):
    total_frames: int
    total_detections: int
    false_positives_suppressed: int
    avg_inference_ms: float
    avg_confidence: float
    avg_consistency: float
    model_loaded: bool
    device: str
    uptime_seconds: float
    detections_per_minute: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float


class GPSPoint(BaseModel):
    lat: float
    lon: float
    altitude: float
    confidence: float
    frame_id: str
    timestamp: datetime
