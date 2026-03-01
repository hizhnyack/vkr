"""Анализ сцены: VLM для время суток/погода, MiDaS/DPT для глубины."""
from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np

from .scene_description import SceneDescription

logger = logging.getLogger(__name__)

# Флаг наличия тяжёлых зависимостей (VLM, depth)
try:
    import torch
    from transformers import pipeline
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    BLIP_AVAILABLE = True
except ImportError:
    BLIP_AVAILABLE = False


class SceneAnalyzer:
    """Анализ выбранных кадров: время суток, погода, глубина."""

    def __init__(
        self,
        vlm_model_name: str = "Salesforce/blip2-opt-2.7b",
        depth_model_name: str = "Intel/dpt-hybrid-midas",
        device: str = "cuda",
        load_vlm_4bit: bool = True,
    ):
        self.vlm_model_name = vlm_model_name
        self.depth_model_name = depth_model_name
        self.device = device
        self.load_vlm_4bit = load_vlm_4bit
        self._vlm_pipeline = None
        self._depth_pipeline = None

    def _get_vlm(self):
        if self._vlm_pipeline is not None:
            return self._vlm_pipeline
        if not TORCH_AVAILABLE or not BLIP_AVAILABLE:
            logger.warning("VLM недоступен (transformers/torch). Возвращаем дефолтное описание сцены.")
            return None
        try:
            from PIL import Image
            # Используем BLIP (легче чем blip2-opt-2.7b для 8GB)
            model_id = "Salesforce/blip-image-captioning-base"
            self._vlm_pipeline = pipeline(
                "image-to-text",
                model=model_id,
                device=0 if self.device == "cuda" else -1,
            )
            return self._vlm_pipeline
        except Exception as e:
            logger.warning("Не удалось загрузить VLM: %s", e)
            return None

    def _get_depth(self):
        if self._depth_pipeline is not None:
            return self._depth_pipeline
        if not TORCH_AVAILABLE:
            return None
        try:
            from transformers import pipeline as hf_pipeline
            self._depth_pipeline = hf_pipeline(
                task="depth-estimation",
                model=self.depth_model_name,
                device=0 if self.device == "cuda" else -1,
            )
            return self._depth_pipeline
        except Exception as e:
            logger.warning("Не удалось загрузить depth: %s", e)
            return None

    def _frame_to_pil(self, frame: np.ndarray):
        from PIL import Image
        if frame.dtype != np.uint8:
            frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8) if frame.max() <= 1 else frame.astype(np.uint8)
        return Image.fromarray(frame)

    def analyze_time_weather(self, frame: np.ndarray) -> Tuple[str, str]:
        """Вернуть (time_of_day, weather) по кадру."""
        pipe = self._get_vlm()
        if pipe is None:
            return "day", "clear"
        try:
            pil = self._frame_to_pil(frame)
            out = pipe(pil)
            caption = out[0]["generated_text"] if isinstance(out, list) and out else str(out)
            caption = caption.lower()
            time_of_day = "day"
            if "night" in caption or "dark" in caption:
                time_of_day = "night"
            elif "dusk" in caption or "sunset" in caption:
                time_of_day = "dusk"
            elif "dawn" in caption or "sunrise" in caption:
                time_of_day = "dawn"
            weather = "clear"
            if "rain" in caption or "rainy" in caption:
                weather = "rain"
            elif "fog" in caption or "foggy" in caption:
                weather = "fog"
            elif "snow" in caption:
                weather = "snow"
            elif "cloud" in caption or "overcast" in caption:
                weather = "overcast"
            return time_of_day, weather
        except Exception as e:
            logger.warning("VLM анализ не удался: %s", e)
            return "day", "clear"

    def estimate_depth(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Оценка глубины, возврат карты (H, W)."""
        pipe = self._get_depth()
        if pipe is None:
            return None
        try:
            pil = self._frame_to_pil(frame)
            out = pipe(pil)
            if isinstance(out, dict) and "depth" in out:
                depth = out["depth"]
            elif isinstance(out, list) and len(out) and hasattr(out[0], "numpy"):
                depth = np.array(out[0])
            else:
                depth = np.array(out)
            if hasattr(depth, "numpy"):
                depth = depth.numpy()
            return np.squeeze(depth)
        except Exception as e:
            logger.warning("Depth не удался: %s", e)
            return None

    def analyze_frames(
        self,
        frames: List[Tuple[int, np.ndarray]],
    ) -> List[SceneDescription]:
        """По списку (frame_index, frame) вернуть список SceneDescription."""
        results = []
        for idx, frame in frames:
            time_of_day, weather = self.analyze_time_weather(frame)
            depth = self.estimate_depth(frame)
            desc = SceneDescription(
                time_of_day=time_of_day,
                weather=weather,
                lighting="natural",
                lighting_direction=None,
                frame_index=idx,
            )
            if depth is not None:
                desc.set_depth_map(depth)
            results.append(desc)
        return results
