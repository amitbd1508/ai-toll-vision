import tempfile
from pathlib import Path

import pytest

from backend.database.db import Database


@pytest.fixture
def db():
    with tempfile.TemporaryDirectory() as tmp:
        yield Database(str(Path(tmp) / "test.db"))


def test_video_crud(db):
    db.insert_video({
        "video_id": "v1", "filename": "a.mp4", "filepath": "/tmp/a.mp4",
        "uploaded_at": "2026-01-01T00:00:00", "duration_sec": 10.0, "fps": 25.0,
        "width": 640, "height": 480, "status": "uploaded",
    })
    video = db.get_video("v1")
    assert video["filename"] == "a.mp4"
    db.update_video_status("v1", "done", annotated_path="/tmp/out.mp4")
    assert db.get_video("v1")["status"] == "done"
    assert len(db.list_videos()) == 1


def test_vehicle_upsert(db):
    db.insert_video({
        "video_id": "v1", "filename": "a.mp4", "filepath": "/tmp/a.mp4",
        "uploaded_at": "2026-01-01T00:00:00", "duration_sec": 10.0, "fps": 25.0,
        "width": 640, "height": 480, "status": "uploaded",
    })
    vehicle = {
        "vehicle_id": "v1-1", "video_id": "v1", "track_id": 1, "vehicle_type": "car",
        "first_seen_sec": 0.0, "last_seen_sec": 1.0, "first_seen_frame": 0,
        "last_seen_frame": 5, "avg_confidence": 0.8, "direction": "entering", "is_active": 1,
    }
    db.upsert_vehicle(vehicle)
    vehicle["last_seen_sec"] = 2.0
    db.upsert_vehicle(vehicle)
    vehicles = db.list_vehicles("v1")
    assert len(vehicles) == 1
    assert vehicles[0]["last_seen_sec"] == 2.0


def test_event_insert_and_filter(db):
    db.insert_video({
        "video_id": "v1", "filename": "a.mp4", "filepath": "/tmp/a.mp4",
        "uploaded_at": "2026-01-01T00:00:00", "duration_sec": 10.0, "fps": 25.0,
        "width": 640, "height": 480, "status": "uploaded",
    })
    db.insert_event({
        "event_id": "e1", "video_id": "v1", "vehicle_id": None, "event_type": "VEHICLE_ENTERED",
        "timestamp_sec": 1.0, "frame_number": 25, "confidence": 0.9, "metadata": {"k": "v"},
    })
    db.insert_event({
        "event_id": "e2", "video_id": "v1", "vehicle_id": None, "event_type": "LOW_CONFIDENCE",
        "timestamp_sec": 2.0, "frame_number": 50, "confidence": 0.2, "metadata": {},
    })
    all_events = db.list_events("v1")
    assert len(all_events) == 2
    assert all_events[0]["metadata"] == {"k": "v"}

    filtered = db.list_events("v1", event_type="LOW_CONFIDENCE")
    assert len(filtered) == 1
    assert filtered[0]["event_id"] == "e2"

    high_conf = db.list_events("v1", min_confidence=0.5)
    assert len(high_conf) == 1
    assert high_conf[0]["event_id"] == "e1"
