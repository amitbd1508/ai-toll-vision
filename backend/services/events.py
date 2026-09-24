"""
Event engine.

Turns tracker state into the structured events defined in the spec:
VEHICLE_ENTERED, VEHICLE_EXITED, VEHICLE_TYPE_CHANGED, PLATE_DETECTED,
OCR_COMPLETED, LOW_CONFIDENCE, UNUSUAL_VISUAL_CONDITION.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Optional

from .tracker import Track

LOW_CONFIDENCE_THRESHOLD = 0.35


@dataclass
class ROI:
    x1: float
    y1: float
    x2: float
    y2: float
    direction_axis: str = "y"  # "y" -> vertical lane, "x" -> horizontal lane

    def contains(self, cx: float, cy: float) -> bool:
        return self.x1 <= cx <= self.x2 and self.y1 <= cy <= self.y2


def estimate_direction(track: Track, roi: Optional[ROI]) -> str:
    """Entering vs exiting from centroid movement along the ROI's axis."""
    history = track.centroid_history
    if len(history) < 2:
        return "unknown"
    axis = roi.direction_axis if roi else "y"
    idx = 2 if axis == "y" else 1  # (frame, cx, cy) -> cx index 1, cy index 2
    start = history[0][idx]
    end = history[-1][idx]
    if abs(end - start) < 2:  # px, essentially static
        return "unknown"
    if axis == "y":
        return "entering" if end > start else "exiting"
    return "entering" if end > start else "exiting"


class EventEngine:
    """Stateful: call `on_frame` once per processed frame with the tracks
    active in that frame. Emits a list of event dicts ready for DB insert."""

    def __init__(self, video_id: str, roi: Optional[ROI] = None):
        self.video_id = video_id
        self.roi = roi
        self._seen_in_roi: set[int] = set()
        self._last_active: set[int] = set()
        self._last_type: dict[int, str] = {}
        self._last_track: dict[int, Track] = {}

    def _vehicle_id(self, track_id: int) -> str:
        return f"{self.video_id}-{track_id}"

    def _new_event(self, event_type: str, vehicle_id: Optional[str], timestamp_sec: float,
                    frame_number: int, confidence: Optional[float], metadata: dict) -> dict:
        return {
            "event_id": str(uuid.uuid4()),
            "video_id": self.video_id,
            "vehicle_id": vehicle_id,
            "event_type": event_type,
            "timestamp_sec": timestamp_sec,
            "frame_number": frame_number,
            "confidence": confidence,
            "metadata": metadata,
        }

    def on_frame(self, tracks: list[Track], frame_number: int, timestamp_sec: float) -> list[dict]:
        events: list[dict] = []
        active_ids = set()

        for track in tracks:
            vehicle_id = self._vehicle_id(track.track_id)
            cx, cy = track.centroid
            in_roi = self.roi.contains(cx, cy) if self.roi else True
            det_conf = track.detection.confidence

            if in_roi:
                active_ids.add(track.track_id)

                if track.track_id not in self._seen_in_roi:
                    self._seen_in_roi.add(track.track_id)
                    events.append(self._new_event(
                        "VEHICLE_ENTERED", vehicle_id, timestamp_sec, frame_number, det_conf,
                        {"vehicle_type": track.majority_type, "source": track.detection.source},
                    ))

                prev_type = self._last_type.get(track.track_id)
                if prev_type and prev_type != track.majority_type:
                    events.append(self._new_event(
                        "VEHICLE_TYPE_CHANGED", vehicle_id, timestamp_sec, frame_number, det_conf,
                        {"from": prev_type, "to": track.majority_type},
                    ))
                self._last_type[track.track_id] = track.majority_type
                self._last_track[track.track_id] = track

                if det_conf < LOW_CONFIDENCE_THRESHOLD:
                    events.append(self._new_event(
                        "LOW_CONFIDENCE", vehicle_id, timestamp_sec, frame_number, det_conf,
                        {"reason": "detection confidence below threshold"},
                    ))

        # Vehicles that were in the ROI last frame but aren't now => exited.
        for tid in self._last_active - active_ids:
            if tid in self._seen_in_roi:
                last_track = self._last_track.get(tid)
                direction = estimate_direction(last_track, self.roi) if last_track else "unknown"
                events.append(self._new_event(
                    "VEHICLE_EXITED", self._vehicle_id(tid), timestamp_sec, frame_number, None,
                    {"direction": direction, "vehicle_type": self._last_type.get(tid, "unknown")},
                ))

        self._last_active = active_ids
        return events

    def unusual_condition_event(self, frame_number: int, timestamp_sec: float,
                                 conditions: list[str], confidence: float) -> dict:
        return self._new_event(
            "UNUSUAL_VISUAL_CONDITION", None, timestamp_sec, frame_number, confidence,
            {"conditions": conditions},
        )
