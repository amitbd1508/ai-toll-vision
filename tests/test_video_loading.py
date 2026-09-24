import tempfile
from pathlib import Path

import cv2
import numpy as np


def _make_synthetic_video(path: str, n_frames=15, size=(320, 240), fps=10):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(n_frames):
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        # Moving square to simulate a vehicle crossing the frame.
        x = 10 + i * 15
        cv2.rectangle(frame, (x, 80), (x + 40, 120), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()


def test_synthetic_video_is_readable():
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "synthetic.mp4")
        _make_synthetic_video(path)
        cap = cv2.VideoCapture(path)
        assert cap.isOpened()
        count = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            count += 1
        cap.release()
        assert count > 0
