"""
Vehicle detector.

Wraps Ultralytics YOLO. If `ultralytics`/`torch` are not installed, and
`demo_mode.allow_mock_detector` is true in config, falls back to a simple
motion-blob "mock detector" (pure OpenCV, no ML) so the rest of the pipeline
(tracking, ROI, events, DB, dashboard) can still be exercised end-to-end.

The mock detector's boxes are NOT real vehicle detections — every detection
it emits is labeled with source="mock" and the API/dashboard must display
that clearly. This module never silently pretends a real model produced a
mock result.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

logger = logging.getLogger("toll_vision.detector")

COCO_VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_name: str
    source: str = "yolo"  # "yolo" or "mock"


class VehicleDetector:
    def __init__(self, model_path: str, confidence: float, device: str,
                 allow_mock: bool = True):
        self.confidence = confidence
        self.device = device
        self.allow_mock = allow_mock
        self.model = None
        self.mode = "uninitialized"
        self._load()

    def _load(self):
        try:
            from ultralytics import YOLO  # requires torch

            self.model = YOLO(self._resolve_model_path())
            self.mode = "yolo"
            logger.info("Loaded YOLO model on device=%s", self.device)
        except Exception as exc:  # noqa: broad — many possible import/download errors
            if not self.allow_mock:
                raise RuntimeError(
                    f"YOLO/ultralytics unavailable and mock detector disabled: {exc}"
                ) from exc
            logger.warning(
                "YOLO unavailable (%s). Falling back to MOCK background-subtraction "
                "detector. Detections will be labeled source='mock' and are NOT real "
                "vehicle recognitions.", exc,
            )
            self.mode = "mock"
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=40, detectShadows=False
            )

    def _resolve_model_path(self) -> str:
        # ultralytics auto-downloads the small pretrained weights by name
        # (e.g. "yolov8n.pt") on first use if not present locally.
        return self.model_path if hasattr(self, "model_path") else "yolov8n.pt"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.mode == "yolo":
            return self._detect_yolo(frame)
        return self._detect_mock(frame)

    def _detect_yolo(self, frame: np.ndarray) -> list[Detection]:
        results = self.model.predict(
            frame, conf=self.confidence, device=self.device, verbose=False
        )
        detections = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in COCO_VEHICLE_CLASSES:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]
                detections.append(
                    Detection(
                        x1=x1, y1=y1, x2=x2, y2=y2,
                        confidence=float(box.conf[0]),
                        class_name=COCO_VEHICLE_CLASSES[cls_id],
                        source="yolo",
                    )
                )
        return detections

    def _detect_mock(self, frame: np.ndarray) -> list[Detection]:
        """Pure-OpenCV moving-blob detector. Good enough to demonstrate the
        pipeline plumbing; NOT a vehicle classifier."""
        fg_mask = self._bg_subtractor.apply(frame)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        fg_mask = cv2.dilate(fg_mask, np.ones((9, 9), np.uint8), iterations=2)
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        frame_area = frame.shape[0] * frame.shape[1]
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < 0.002 * frame_area:  # ignore tiny noise blobs
                continue
            x, y, w, h = cv2.boundingRect(c)
            detections.append(
                Detection(
                    x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h),
                    confidence=0.5,  # arbitrary — mock has no real confidence signal
                    class_name="other",
                    source="mock",
                )
            )
        return detections
