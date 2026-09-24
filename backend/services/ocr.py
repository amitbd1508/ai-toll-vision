"""
License plate OCR — optional, privacy-sensitive stage.

Disabled by default. This module does NOT include a license-plate
*detector* (locating the plate within the vehicle crop) — that requires a
dedicated model not bundled with this POC. Per the spec: "If no reliable
plate detector/model is available, gracefully disable this feature rather
than inventing results." So:

  - `plate_detector_available` in config.yaml defaults to False.
  - Even if OCR engines (EasyOCR/PaddleOCR) are installed, this module will
    refuse to run without an explicit plate-region source, and every result
    it does produce is tagged `is_experimental=True`.
  - Raw plate image crops are never persisted unless
    `privacy.store_raw_plate_crops` is explicitly set true.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("toll_vision.ocr")


class PlateOCR:
    def __init__(self, enabled: bool, engine: str, plate_detector_available: bool):
        self.enabled = enabled
        self.engine_name = engine
        self.plate_detector_available = plate_detector_available
        self.available = False
        self.reader = None
        if enabled:
            self._load()

    def _load(self):
        if not self.plate_detector_available:
            logger.warning(
                "OCR is enabled in config but no plate detector is configured. "
                "Skipping OCR stage entirely rather than guessing plate regions."
            )
            return
        try:
            if self.engine_name == "easyocr":
                import easyocr  # type: ignore

                self.reader = easyocr.Reader(["en"], gpu=False)
            elif self.engine_name == "paddleocr":
                from paddleocr import PaddleOCR  # type: ignore

                self.reader = PaddleOCR(use_angle_cls=True, lang="en")
            else:
                raise ValueError(f"Unknown OCR engine: {self.engine_name}")
            self.available = True
            logger.info("OCR engine '%s' loaded", self.engine_name)
        except Exception as exc:  # noqa: broad
            logger.warning("OCR engine unavailable (%s). OCR stage will be skipped.", exc)
            self.available = False

    def read_plate(self, plate_crop: np.ndarray) -> tuple[str | None, float]:
        """Returns (plate_text, confidence). (None, 0.0) if unavailable."""
        if not self.available:
            return None, 0.0
        if self.engine_name == "easyocr":
            results = self.reader.readtext(plate_crop)
            if not results:
                return None, 0.0
            _, text, conf = max(results, key=lambda r: r[2])
            return text, float(conf)
        else:  # paddleocr
            results = self.reader.ocr(plate_crop, cls=True)
            if not results or not results[0]:
                return None, 0.0
            _, (text, conf) = max(results[0], key=lambda r: r[1][1])
            return text, float(conf)
