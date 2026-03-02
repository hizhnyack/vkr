# Generation: Stable Diffusion + ControlNet
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    from PIL import Image
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


class ContentGenerator:
    """Generate images from prompt + depth conditioning."""

    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_id: str = "lllyasviel/control_v11f1p_sd15_depth",
        device: str = "cuda",
        num_inference_steps: int = 30,
        guidance_scale: float = 7.5,
        max_resolution: int = 512,
    ):
        self.model_id = model_id
        self.controlnet_id = controlnet_id
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.max_resolution = max_resolution
        self._pipe = None

    def _get_pipeline(self):
        if self._pipe is not None:
            return self._pipe
        if not DIFFUSERS_AVAILABLE:
            raise RuntimeError("diffusers not installed")
        controlnet = ControlNetModel.from_pretrained(self.controlnet_id, torch_dtype=torch.float16)
        self._pipe = StableDiffusionControlNetPipeline.from_pretrained(
            self.model_id,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        self._pipe.scheduler = UniPCMultistepScheduler.from_config(self._pipe.scheduler.config)
        self._pipe = self._pipe.to(self.device)
        return self._pipe

    def _depth_to_control_image(self, depth: np.ndarray, bbox: tuple):
        from PIL import Image
        x1, y1, x2, y2 = bbox
        crop = depth[y1:y2, x1:x2]
        if crop.size == 0:
            crop = np.zeros((64, 64), dtype=np.float32)
        if crop.max() > crop.min():
            crop = (crop - crop.min()) / (crop.max() - crop.min()) * 255
        else:
            crop = np.zeros_like(crop) + 128
        crop = np.clip(crop, 0, 255).astype(np.uint8)
        crop = np.stack([crop, crop, crop], axis=-1)
        return Image.fromarray(crop)

    def generate_for_region(
        self,
        prompt: str,
        frame_rgb: np.ndarray,
        bbox: tuple,
        depth_map: Optional[np.ndarray] = None,
        time_of_day: str = "day",
        weather: str = "clear",
        negative_prompt: str = "blurry, low quality",
    ) -> np.ndarray:
        if not DIFFUSERS_AVAILABLE:
            x1, y1, x2, y2 = bbox
            patch = frame_rgb[y1:y2, x1:x2].copy()
            return np.clip(patch.astype(np.float32) * 0.7 + 30, 0, 255).astype(np.uint8)
        pipe = self._get_pipeline()
        x1, y1, x2, y2 = bbox
        if depth_map is None:
            depth_map = np.ones((frame_rgb.shape[0], frame_rgb.shape[1]), dtype=np.float32) * 0.5
        control_image = self._depth_to_control_image(depth_map, bbox)
        control_image = control_image.resize((min(512, x2 - x1), min(512, y2 - y1)))
        full_prompt = f"{prompt}, {time_of_day}, {weather}, realistic"
        with torch.inference_mode():
            out = pipe(
                full_prompt,
                image=control_image,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                negative_prompt=negative_prompt,
            )
        image = out.images[0]
        img_np = np.array(image)
        from PIL import Image
        target_h, target_w = y2 - y1, x2 - x1
        if img_np.shape[0] != target_h or img_np.shape[1] != target_w:
            image = Image.fromarray(img_np).resize((target_w, target_h), Image.Resampling.LANCZOS)
            img_np = np.array(image)
        return img_np
