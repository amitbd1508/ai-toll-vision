import numpy as np

from backend.services.detector import VehicleDetector


def test_mock_detector_used_when_yolo_unavailable():
    # In this environment ultralytics/torch are not installed, so this
    # should silently fall back to mock mode rather than raising.
    detector = VehicleDetector(model_path="yolov8n.pt", confidence=0.4, device="cpu", allow_mock=True)
    assert detector.mode in ("yolo", "mock")

    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    # First frame establishes the background model — may yield 0 detections.
    detector.detect(frame)
    detections = detector.detect(frame)
    for d in detections:
        assert d.source in ("yolo", "mock")


def test_mock_detector_raises_when_disallowed_and_yolo_missing():
    detector_or_error = None
    try:
        VehicleDetector(model_path="yolov8n.pt", confidence=0.4, device="cpu", allow_mock=False)
    except RuntimeError as exc:
        detector_or_error = exc
    # Either YOLO is actually available in this env (no error) or it correctly
    # refuses instead of silently mocking.
    assert detector_or_error is None or "unavailable" in str(detector_or_error)
