from __future__ import annotations

import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import cv2
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse

from ..config import load_config
from ..database.db import Database
from ..models.schemas import HealthOut, ProcessRequest, ProcessResult, VideoOut
from ..services.device import detect_device
from ..services.video_processor import VideoProcessor

router = APIRouter()

_config = load_config()
_db = Database(_config["database"]["path"])
_processor: VideoProcessor | None = None  # lazily created, loaded once, reused


def get_processor() -> VideoProcessor:
    global _processor
    if _processor is None:
        _processor = VideoProcessor(_config, _db)
    return _processor


ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov"}


@router.get("/health", response_model=HealthOut)
def health():
    device_info = detect_device(_config["device"].get("prefer", "auto"))
    stages = {
        "detection": "yolo" if get_processor().detector.mode == "yolo" else "mock (no torch/ultralytics found)",
        "tracking": "enabled",
        "segmentation": get_processor().segmenter.available,
        "ocr": get_processor().ocr.available,
        "vlm": get_processor().vlm.available,
    }
    return HealthOut(status="ok", device=device_info, stages=stages)


@router.post("/videos/upload", response_model=VideoOut)
async def upload_video(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type {ext}. Use one of {ALLOWED_EXTENSIONS}")

    video_id = str(uuid.uuid4())
    videos_dir = Path(_config["paths"]["videos_dir"])
    videos_dir.mkdir(parents=True, exist_ok=True)
    dest = videos_dir / f"{video_id}{ext}"

    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)

    cap = cv2.VideoCapture(str(dest))
    if not cap.isOpened():
        dest.unlink(missing_ok=True)
        raise HTTPException(400, "Could not read uploaded video — file may be corrupt")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps else None
    cap.release()

    _db.insert_video({
        "video_id": video_id, "filename": file.filename, "filepath": str(dest),
        "uploaded_at": datetime.now(timezone.utc).isoformat(), "duration_sec": duration,
        "fps": fps, "width": width, "height": height, "status": "uploaded",
    })
    return VideoOut(**_db.get_video(video_id))


@router.post("/videos/{video_id}/process", response_model=ProcessResult)
def process_video(video_id: str, req: ProcessRequest = ProcessRequest()):
    video = _db.get_video(video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    try:
        result = get_processor().process(video_id, roi_rect=req.roi)
    except Exception as exc:
        raise HTTPException(500, f"Processing failed: {exc}") from exc
    return ProcessResult(**result)


@router.get("/videos", response_model=list[VideoOut])
def list_videos():
    return [VideoOut(**v) for v in _db.list_videos()]


@router.get("/videos/{video_id}", response_model=VideoOut)
def get_video(video_id: str):
    video = _db.get_video(video_id)
    if not video:
        raise HTTPException(404, "Video not found")
    return VideoOut(**video)


@router.get("/videos/{video_id}/vehicles")
def get_vehicles(video_id: str):
    if not _db.get_video(video_id):
        raise HTTPException(404, "Video not found")
    return _db.list_vehicles(video_id)


@router.get("/videos/{video_id}/events")
def get_events(video_id: str, event_type: str | None = None,
                min_confidence: float | None = None,
                start_sec: float | None = None, end_sec: float | None = None):
    if not _db.get_video(video_id):
        raise HTTPException(404, "Video not found")
    return _db.list_events(video_id, event_type, min_confidence, start_sec, end_sec)


@router.get("/videos/{video_id}/analysis")
def get_analysis(video_id: str):
    if not _db.get_video(video_id):
        raise HTTPException(404, "Video not found")
    return {
        "vehicles": _db.list_vehicles(video_id),
        "events": _db.list_events(video_id),
        "ocr_results": _db.list_ocr_results(video_id),
        "vlm_analysis": _db.list_vlm_analysis(video_id),
    }
