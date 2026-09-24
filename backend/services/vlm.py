"""
Vision-language model reasoning — optional stage.

Disabled by default (Qwen2.5-VL-7B is a multi-GB download and needs a real
GPU to run at usable speed). When enabled, only a *sample* of frames/events
are sent to the VLM (config: vlm.sample_every_n_events, vlm.max_frames_per_video)
— never every frame, per the performance requirements.

The prompt instructs the model to report only observable facts and return
strict JSON; if the model's output isn't valid JSON we don't guess — we
return an "unparseable" result explicitly.
"""
from __future__ import annotations

import json
import logging

import numpy as np

logger = logging.getLogger("toll_vision.vlm")

SYSTEM_PROMPT = (
    "You are analyzing a toll-booth camera frame. Describe only observable "
    "facts. Identify: 1) number of visible vehicles, 2) vehicle types, "
    "3) lane occupancy, 4) unusual visual conditions, 5) whether the image "
    "is unclear. Do not infer identity, intent, emotion, or criminal "
    "behavior. Return ONLY valid JSON, no other text, in exactly this shape: "
    '{"vehicle_count": int, "vehicle_types": [string], "lane_occupied": bool, '
    '"visual_conditions": [string], "is_unclear": bool, "confidence": float}'
)


class VLMReasoner:
    def __init__(self, enabled: bool, model_name: str, device: str):
        self.enabled = enabled
        self.model_name = model_name
        self.device = device
        self.available = False
        self.model = None
        self.processor = None
        if enabled:
            self._load()

    def _load(self):
        try:
            from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
            import torch

            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                self.model_name, torch_dtype=torch.float16 if self.device != "cpu" else torch.float32
            ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.available = True
            logger.info("VLM '%s' loaded on %s", self.model_name, self.device)
        except Exception as exc:  # noqa: broad
            logger.warning(
                "VLM unavailable (%s). VLM reasoning stage will be skipped.", exc
            )
            self.available = False

    def analyze_frame(self, frame_rgb: np.ndarray) -> dict:
        """Returns a structured dict matching SYSTEM_PROMPT's JSON schema, or
        a dict with error info if the VLM isn't available / output isn't
        parseable. Never fabricates values."""
        if not self.available:
            return {"error": "vlm_unavailable"}

        from PIL import Image

        image = Image.fromarray(frame_rgb)
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": SYSTEM_PROMPT}],
        }]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors="pt").to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=256)
        raw = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        try:
            # Model may echo the prompt; take the last JSON-looking substring.
            start = raw.rfind("{")
            end = raw.rfind("}") + 1
            parsed = json.loads(raw[start:end])
            parsed["_raw"] = raw
            return parsed
        except Exception:
            logger.warning("VLM output was not parseable JSON")
            return {"error": "unparseable_output", "_raw": raw}
