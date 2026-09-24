from backend.services.detector import Detection
from backend.services.events import ROI, EventEngine, estimate_direction
from backend.services.tracker import IOUTracker


def make_det(x1, y1, x2, y2, conf=0.8, cls="car"):
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf, class_name=cls)


def test_tracker_assigns_and_maintains_ids():
    tracker = IOUTracker(iou_threshold=0.3, max_age_frames=5)

    tracks = tracker.update([make_det(10, 10, 50, 50)], frame_number=0)
    assert len(tracks) == 1
    tid = tracks[0].track_id

    # Same vehicle, slightly moved -> same track id.
    tracks = tracker.update([make_det(15, 10, 55, 50)], frame_number=1)
    assert len(tracks) == 1
    assert tracks[0].track_id == tid


def test_tracker_drops_stale_tracks():
    tracker = IOUTracker(iou_threshold=0.3, max_age_frames=2)
    tracker.update([make_det(10, 10, 50, 50)], frame_number=0)
    tracker.update([], frame_number=1)
    tracker.update([], frame_number=2)
    tracker.update([], frame_number=3)  # exceeds max_age_frames -> dropped
    assert len(tracker.tracks) == 0


def test_roi_crossing_generates_enter_and_exit_events():
    roi = ROI(x1=0, y1=0, x2=100, y2=100, direction_axis="y")
    engine = EventEngine("video1", roi)
    tracker = IOUTracker(iou_threshold=0.2, max_age_frames=5)

    # Frame 0: vehicle appears inside ROI.
    tracks = tracker.update([make_det(10, 10, 30, 30)], frame_number=0)
    events = engine.on_frame(tracks, 0, 0.0)
    assert any(e["event_type"] == "VEHICLE_ENTERED" for e in events)

    # Frame 1: vehicle still inside, moving.
    tracks = tracker.update([make_det(10, 20, 30, 40)], frame_number=1)
    engine.on_frame(tracks, 1, 0.2)

    # Frame 2: vehicle leaves ROI entirely (detection far outside).
    tracks = tracker.update([make_det(200, 200, 230, 230)], frame_number=2)
    events = engine.on_frame(tracks, 2, 0.4)
    assert any(e["event_type"] == "VEHICLE_EXITED" for e in events)


def test_low_confidence_event_emitted():
    roi = None
    engine = EventEngine("video1", roi)
    tracker = IOUTracker()
    tracks = tracker.update([make_det(0, 0, 20, 20, conf=0.1)], frame_number=0)
    events = engine.on_frame(tracks, 0, 0.0)
    assert any(e["event_type"] == "LOW_CONFIDENCE" for e in events)


def test_estimate_direction_from_centroid_movement():
    tracker = IOUTracker(iou_threshold=0.1)
    tracker.update([make_det(0, 0, 20, 20)], frame_number=0)
    # Small enough shift to still overlap (same track), repeated to build history.
    tracker.update([make_det(0, 8, 20, 28)], frame_number=1)
    tracks = tracker.update([make_det(0, 16, 20, 36)], frame_number=2)
    assert len(tracks) == 1  # confirms it's still the same track, not a new one
    roi = ROI(0, 0, 100, 100, direction_axis="y")
    direction = estimate_direction(tracks[0], roi)
    assert direction in ("entering", "exiting")
