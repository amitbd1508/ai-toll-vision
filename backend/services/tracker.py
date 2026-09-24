"""
Multi-object tracker.

For real YOLO detections, Ultralytics ships ByteTrack/BoT-SORT built in
(`model.track(..., tracker="bytetrack.yaml")`) — see detector.py docstring
for how to switch to it. This module implements a small, dependency-free
IOU-matching tracker so that:
  (a) tracking works identically for the mock detector too (no torch needed
      to demo the tracking/ROI/event pipeline), and
  (b) the tracking logic is transparent and easy to read for a POC.

It is a simplified SORT: greedy IOU matching + track aging, no Kalman
filter. That's a deliberate scope cut for the POC — swap in ByteTrack for
production-grade tracking.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .detector import Detection


def iou(a: Detection, b: Detection) -> float:
    xx1, yy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    xx2, yy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    if inter == 0:
        return 0.0
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / (area_a + area_b - inter)


@dataclass
class Track:
    track_id: int
    detection: Detection
    age_since_seen: int = 0
    hits: int = 1
    confidences: list = field(default_factory=list)
    centroid_history: list = field(default_factory=list)  # [(frame_no, cx, cy)]
    vehicle_type_votes: dict = field(default_factory=dict)

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.detection.x1 + self.detection.x2) / 2,
                 (self.detection.y1 + self.detection.y2) / 2)

    @property
    def majority_type(self) -> str:
        if not self.vehicle_type_votes:
            return self.detection.class_name
        return max(self.vehicle_type_votes, key=self.vehicle_type_votes.get)


class IOUTracker:
    def __init__(self, iou_threshold: float = 0.3, max_age_frames: int = 30):
        self.iou_threshold = iou_threshold
        self.max_age_frames = max_age_frames
        self.tracks: dict[int, Track] = {}
        self._next_id = 1

    def update(self, detections: list[Detection], frame_number: int) -> list[Track]:
        unmatched_dets = list(range(len(detections)))
        matched_track_ids = set()

        # Greedy IOU matching, highest overlap first.
        pairs = []
        for tid, track in self.tracks.items():
            for di, det in enumerate(detections):
                score = iou(track.detection, det)
                if score >= self.iou_threshold:
                    pairs.append((score, tid, di))
        pairs.sort(reverse=True, key=lambda p: p[0])

        used_dets = set()
        for score, tid, di in pairs:
            if tid in matched_track_ids or di in used_dets:
                continue
            track = self.tracks[tid]
            det = detections[di]
            track.detection = det
            track.age_since_seen = 0
            track.hits += 1
            track.confidences.append(det.confidence)
            track.vehicle_type_votes[det.class_name] = track.vehicle_type_votes.get(
                det.class_name, 0
            ) + 1
            cx, cy = track.centroid
            track.centroid_history.append((frame_number, cx, cy))
            matched_track_ids.add(tid)
            used_dets.add(di)

        unmatched_dets = [i for i in range(len(detections)) if i not in used_dets]

        # Age out unmatched tracks; drop if too stale.
        for tid in list(self.tracks.keys()):
            if tid not in matched_track_ids:
                self.tracks[tid].age_since_seen += 1
                if self.tracks[tid].age_since_seen > self.max_age_frames:
                    del self.tracks[tid]

        # New tracks for unmatched detections.
        for di in unmatched_dets:
            det = detections[di]
            track = Track(track_id=self._next_id, detection=det)
            track.confidences.append(det.confidence)
            track.vehicle_type_votes[det.class_name] = 1
            cx, cy = track.centroid
            track.centroid_history.append((frame_number, cx, cy))
            self.tracks[self._next_id] = track
            self._next_id += 1

        return [t for t in self.tracks.values() if t.age_since_seen == 0]

    def active_tracks(self) -> list[Track]:
        return [t for t in self.tracks.values() if t.age_since_seen == 0]
