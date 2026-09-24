"""Detect the best available compute device at startup, once, and reuse it."""
from __future__ import annotations

import logging

logger = logging.getLogger("toll_vision.device")

_DEVICE_CACHE: dict | None = None


def detect_device(prefer: str = "auto") -> dict:
    """Return {"device": "cuda"|"mps"|"cpu", "name": str, "reason": str}.

    Cached after first call so we don't re-probe torch/CUDA repeatedly.
    """
    global _DEVICE_CACHE
    if _DEVICE_CACHE is not None:
        return _DEVICE_CACHE

    result = {"device": "cpu", "name": "CPU", "reason": "default fallback"}

    try:
        import torch  # noqa: local import — torch may not be installed

        if prefer in ("auto", "cuda") and torch.cuda.is_available():
            result = {
                "device": "cuda",
                "name": torch.cuda.get_device_name(0),
                "reason": "CUDA GPU detected",
            }
        elif prefer in ("auto", "mps") and getattr(torch.backends, "mps", None) \
                and torch.backends.mps.is_available():
            result = {"device": "mps", "name": "Apple Silicon (MPS)", "reason": "MPS detected"}
        else:
            result = {"device": "cpu", "name": "CPU", "reason": "no GPU/MPS detected"}
    except ImportError:
        result = {"device": "cpu", "name": "CPU", "reason": "torch not installed"}
    except Exception as exc:  # noqa: broad — e.g. a corrupted/partial torch install
        logger.warning("torch import/probe failed (%s); falling back to CPU", exc)
        result = {"device": "cpu", "name": "CPU", "reason": f"torch probe failed: {exc}"}

    logger.info("Selected device: %s (%s) — %s", result["device"], result["name"], result["reason"])
    _DEVICE_CACHE = result
    return result
