from __future__ import annotations

from typing import Optional

from pydantic import BaseModel


class VideoOut(BaseModel):
    video_id: str
    filename: str
    status: str
    duration_sec: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    annotated_path: Optional[str] = None
    error_message: Optional[str] = None


class ProcessRequest(BaseModel):
    roi: Optional[list[float]] = None  # [x1, y1, x2, y2]


class ProcessResult(BaseModel):
    video_id: str
    frames_processed: int
    elapsed_sec: float
    annotated_path: str
    detector_mode: str
    device: str


class HealthOut(BaseModel):
    status: str
    device: dict
    stages: dict
