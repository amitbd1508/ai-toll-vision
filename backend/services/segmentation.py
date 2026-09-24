"""
SAM2 segmentation — optional stage.

Disabled by default (config.yaml segmentation.enabled: false) because the
checkpoint + `sam2` package are a heavy install for a POC. When enabled,
this wraps SAM2's automatic/box-prompted predictor. When the package isn't
installed, `available` is False and callers should skip segmentation
entirely rather than fabricate a mask.

Called selectively (high-confidence detections only, capped per video) —
never per-frame — per the performance requirements in the spec.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("toll_vision.segmentation")


class Segmenter:
    def __init__(self, enabled: bool, model_name: str, device: str):
        self.enabled = enabled
        self.device = device
        self.model_name = model_name
        self.available = False
        self.predictor = None
        if enabled:
            self._load()

    def _load(self):
        try:
            from sam2.build_sam import build_sam2  # type: ignore
            from sam2.sam2_image_predictor import SAM2ImagePredictor  # type: ignore

            # Checkpoint/config resolution intentionally left to the user's
            # local SAM2 install docs — path varies by release.
            sam2_model = build_sam2(self.model_name, device=self.device)
            self.predictor = SAM2ImagePredictor(sam2_model)
            self.available = True
            logger.info("SAM2 loaded (%s) on %s", self.model_name, self.device)
        except Exception as exc:  # noqa: broad
            logger.warning(
                "SAM2 not available (%s). Segmentation stage will be skipped — "
                "no masks will be fabricated.", exc,
            )
            self.available = False

    def segment_box(self, frame: np.ndarray, box: tuple[float, float, float, float]):
        """Return a boolean mask for the given (x1,y1,x2,y2) box prompt, or
        None if segmentation isn't available."""
        if not self.available:
            return None
        self.predictor.set_image(frame)
        masks, scores, _ = self.predictor.predict(
            box=np.array(box), multimask_output=False
        )
        return masks[0].astype(bool)
