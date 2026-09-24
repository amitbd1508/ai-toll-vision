"""
Video processing orchestrator.

Ties together detector -> tracker -> event engine -> (optional) segmentation
/ OCR / VLM -> SQLite -> annotated output video. Models are loaded once per
VideoProcessor instance and reused across frames (never reloaded per frame).
"""
from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from ..database.db import Database
from .detector import VehicleDetector
from .device import detect_device
from .events import ROI, EventEngine
from .ocr import PlateOCR
from .segmentation import Segmenter
from .tracker import IOUTracker
from .vlm import VLMReasoner

logger = logging.getLogger("toll_vision.pipeline")

TYPE_COLORS = {
    "car": (60, 180, 75), "truck": (245, 130, 48), "bus": (230, 25, 75),
    "motorcycle": (0, 130, 200), "other": (145, 30, 180),
}


class VideoProcessor:
    def __init__(self, config: dict, db: Database):
        self.config = config
        self.db = db
        self.device_info = detect_device(config.get("device", {}).get("prefer", "auto"))
        self.device = self.device_info["device"]

        det_cfg = config["detection"]
        self.detector = VehicleDetector(
            model_path=det_cfg.get("model", "yolov8n.pt"),
            confidence=det_cfg.get("confidence", 0.4),
            device=self.device,
            allow_mock=config.get("demo_mode", {}).get("allow_mock_detector", True),
        )

        seg_cfg = config["segmentation"]
        self.segmenter = Segmenter(seg_cfg.get("enabled", False), seg_cfg.get("model", ""), self.device)

        ocr_cfg = config["ocr"]
        self.ocr = PlateOCR(
            ocr_cfg.get("enabled", False), ocr_cfg.get("engine", "easyocr"),
            ocr_cfg.get("plate_detector_available", False),
        )

        vlm_cfg = config["vlm"]
        self.vlm = VLMReasoner(vlm_cfg.get("enabled", False), vlm_cfg.get("model", ""), self.device)

        logger.info(
            "Pipeline ready — device=%s detector=%s segmentation=%s ocr=%s vlm=%s",
            self.device, self.detector.mode, self.segmenter.available,
            self.ocr.available, self.vlm.available,
        )

    # ------------------------------------------------------------------
    def process(self, video_id: str, roi_rect: list | None = None) -> dict:
        video_row = self.db.get_video(video_id)
        if not video_row:
            raise ValueError(f"Unknown video_id {video_id}")

        self.db.update_video_status(video_id, "processing")
        t0 = time.time()

        cap = cv2.VideoCapture(video_row["filepath"])
        if not cap.isOpened():
            self.db.update_video_status(video_id, "failed", error_message="Could not open video file")
            raise RuntimeError("Could not open video file")

        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        processing_fps = self.config["video"].get("processing_fps", 5)
        frame_interval = max(1, round(src_fps / processing_fps))
        max_dim = self.config["video"].get("max_dimension", 960)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        scale = min(1.0, max_dim / max(width, height)) if max(width, height) > max_dim else 1.0
        out_w, out_h = int(width * scale), int(height * scale)

        roi = None
        cfg_roi = roi_rect or self.config.get("roi", {}).get("lane_rect")
        if cfg_roi:
            roi = ROI(*cfg_roi, direction_axis=self.config["roi"].get("direction_axis", "y"))

        tracker = IOUTracker(max_age_frames=self.config["tracking"].get("max_age_frames", 30))
        event_engine = EventEngine(video_id, roi)

        outputs_dir = Path(self.config["paths"]["outputs_dir"])
        outputs_dir.mkdir(parents=True, exist_ok=True)
        annotated_path = outputs_dir / f"{video_id}_annotated.mp4"
        writer = cv2.VideoWriter(
            str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"), processing_fps, (out_w, out_h)
        )

        frame_number = 0
        processed_count = 0
        seg_budget = self.config["segmentation"].get("max_per_video", 40)
        seg_used = 0
        vlm_frame_count = 0
        vlm_every = self.config["vlm"].get("sample_every_n_events", 5)
        vlm_max = self.config["vlm"].get("max_frames_per_video", 20)
        event_counter = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_number % frame_interval != 0:
                frame_number += 1
                continue

            if scale != 1.0:
                frame = cv2.resize(frame, (out_w, out_h))

            timestamp_sec = frame_number / src_fps
            detections = self.detector.detect(frame)
            tracks = tracker.update(detections, frame_number)
            frame_events = event_engine.on_frame(tracks, frame_number, timestamp_sec)

            for ev in frame_events:
                if ev["vehicle_id"]:
                    self._upsert_vehicle_from_track(video_id, tracks, ev, roi)
                self.db.insert_event(ev)
                event_counter += 1

            # Optional: segmentation on a capped number of high-confidence tracks.
            masks_by_track = {}
            if self.segmenter.available and seg_used < seg_budget:
                min_conf = self.config["segmentation"].get("min_confidence", 0.75)
                for t in tracks:
                    if seg_used >= seg_budget:
                        break
                    if t.detection.confidence >= min_conf:
                        box = (t.detection.x1, t.detection.y1, t.detection.x2, t.detection.y2)
                        mask = self.segmenter.segment_box(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), box)
                        if mask is not None:
                            masks_by_track[t.track_id] = mask
                            seg_used += 1

            # Optional: VLM reasoning, sampled.
            if self.vlm.available and vlm_frame_count < vlm_max and event_counter and \
                    event_counter % vlm_every == 0:
                result = self.vlm.analyze_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if "error" not in result:
                    self.db.insert_vlm_analysis({
                        "analysis_id": str(uuid.uuid4()), "video_id": video_id,
                        "frame_number": frame_number, "timestamp_sec": timestamp_sec,
                        "vehicle_count": result.get("vehicle_count"),
                        "vehicle_types": result.get("vehicle_types", []),
                        "lane_occupied": int(bool(result.get("lane_occupied", False))),
                        "visual_conditions": result.get("visual_conditions", []),
                        "is_unclear": int(bool(result.get("is_unclear", False))),
                        "confidence": result.get("confidence"),
                        "raw_response": result.get("_raw", ""),
                    })
                    if result.get("is_unclear") or result.get("visual_conditions"):
                        self.db.insert_event(event_engine.unusual_condition_event(
                            frame_number, timestamp_sec,
                            result.get("visual_conditions", []), result.get("confidence", 0.0),
                        ))
                vlm_frame_count += 1

            annotated = self._annotate_frame(frame, tracks, roi, masks_by_track)
            writer.write(annotated)
            processed_count += 1
            frame_number += 1

        cap.release()
        writer.release()

        self.db.update_video_status(video_id, "done", annotated_path=str(annotated_path))
        elapsed = time.time() - t0
        logger.info("Processed video %s: %d frames in %.1fs", video_id, processed_count, elapsed)
        return {
            "video_id": video_id, "frames_processed": processed_count,
            "elapsed_sec": elapsed, "annotated_path": str(annotated_path),
            "detector_mode": self.detector.mode, "device": self.device,
        }

    # ------------------------------------------------------------------
    def _upsert_vehicle_from_track(self, video_id, tracks, ev, roi):
        track = next((t for t in tracks if f"{video_id}-{t.track_id}" == ev["vehicle_id"]), None)
        if not track:
            return
        from .events import estimate_direction

        avg_conf = sum(track.confidences) / len(track.confidences) if track.confidences else 0.0
        first_frame = track.centroid_history[0][0] if track.centroid_history else ev["frame_number"]
        self.db.upsert_vehicle({
            "vehicle_id": ev["vehicle_id"], "video_id": video_id, "track_id": track.track_id,
            "vehicle_type": track.majority_type,
            # first_seen_sec/first_seen_frame are only used on the *first* insert for this
            # vehicle_id (see Database.upsert_vehicle) — later calls only update last_seen/etc.
            "first_seen_sec": ev["timestamp_sec"],
            "last_seen_sec": ev["timestamp_sec"],
            "first_seen_frame": first_frame,
            "last_seen_frame": ev["frame_number"],
            "avg_confidence": avg_conf,
            "direction": estimate_direction(track, roi),
            "is_active": 1,
        })

    def _annotate_frame(self, frame, tracks, roi, masks_by_track) -> np.ndarray:
        out = frame.copy()
        if roi:
            cv2.rectangle(out, (int(roi.x1), int(roi.y1)), (int(roi.x2), int(roi.y2)), (255, 255, 0), 2)
            cv2.putText(out, "LANE ROI", (int(roi.x1), int(roi.y1) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        for t in tracks:
            d = t.detection
            color = TYPE_COLORS.get(t.majority_type, (200, 200, 200))
            p1, p2 = (int(d.x1), int(d.y1)), (int(d.x2), int(d.y2))
            cv2.rectangle(out, p1, p2, color, 2)
            label = f"#{t.track_id} {t.majority_type} {d.confidence:.2f}"
            if d.source == "mock":
                label += " [MOCK]"
            cv2.putText(out, label, (p1[0], max(0, p1[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if t.track_id in masks_by_track:
                mask = masks_by_track[t.track_id]
                overlay = out.copy()
                overlay[mask] = color
                out = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)

        cv2.putText(out, time.strftime("%Y-%m-%d %H:%M:%S"), (10, out.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return out
