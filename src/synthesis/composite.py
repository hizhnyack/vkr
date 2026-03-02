"""Композитинг: вставка сгенерированных патчей в кадры, color matching."""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from .planner import InsertionEvent

logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


def _draw_bbox_border(out: np.ndarray, x1: int, y1: int, x2: int, y2: int, color: tuple, thickness: int) -> None:
    """Отрисовать рамку по границе bbox (для отладки). Изменяет out на месте."""
    h, w = out.shape[0], out.shape[1]
    t = min(thickness, (x2 - x1) // 2, (y2 - y1) // 2)
    if t <= 0:
        return
    # Внешние границы bbox
    y1a, y2a = max(0, y1), min(h, y2)
    x1a, x2a = max(0, x1), min(w, x2)
    c = np.array(color, dtype=out.dtype)
    if out.ndim == 3:
        c = c.reshape(1, 1, -1)
    out[y1a : y1a + t, x1a:x2a] = c
    out[y2a - t : y2a, x1a:x2a] = c
    out[y1a:y2a, x1a : x1a + t] = c
    out[y1a:y2a, x2a - t : x2a] = c


def _color_match_region(source: np.ndarray, target_region: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Подогнать среднее и std source к target_region в области mask."""
    if source.shape != target_region.shape:
        return source
    m = mask.astype(np.float32) / 255.0
    if m.ndim == 2:
        m = m[:, :, np.newaxis]
    for c in range(3):
        ts = target_region[:, :, c].astype(np.float32)
        ss = source[:, :, c].astype(np.float32)
        t_mean = np.mean(ts[m[:, :, 0] > 0.5]) if np.any(m > 0.5) else np.mean(ts)
        t_std = np.std(ts) + 1e-6
        s_mean = np.mean(ss)
        s_std = np.std(ss) + 1e-6
        ss = (ss - s_mean) * (t_std / s_std) + t_mean
        source[:, :, c] = np.clip(ss, 0, 255)
    return source.astype(np.uint8)


class CompositePipeline:
    """Наложение сгенерированного контента на кадры по маскам/bbox с опциональным color matching."""

    def __init__(
        self,
        use_color_matching: bool = True,
        debug_draw_border: bool = False,
        debug_border_color: tuple = (255, 0, 0),
        debug_border_thickness: int = 3,
    ):
        self.use_color_matching = use_color_matching
        self.debug_draw_border = debug_draw_border
        self.debug_border_color = debug_border_color
        self.debug_border_thickness = debug_border_thickness

    def composite_frame(
        self,
        frame: np.ndarray,
        insertion_events: List[InsertionEvent],
        frame_index: int,
        generated_patches: List[tuple],
    ) -> np.ndarray:
        """
        frame — текущий кадр (H,W,3).
        insertion_events — список событий вставки.
        frame_index — номер кадра.
        generated_patches — список (event_index, patch_rgb) для этого кадра.
        Возвращает кадр с наложенными патчами.
        """
        out = frame.astype(np.float32)
        for (ev_idx, patch) in generated_patches:
            if ev_idx >= len(insertion_events):
                continue
            ev = insertion_events[ev_idx]
            if not (ev.start_frame <= frame_index < ev.end_frame):
                continue
            local_idx = frame_index - ev.start_frame
            if ev.masks and local_idx < len(ev.masks):
                mask = ev.masks[local_idx]
            elif ev.bboxes and local_idx < len(ev.bboxes):
                x1, y1, x2, y2 = ev.bboxes[local_idx]
                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                if HAS_CV2:
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
                else:
                    mask[y1:y2, x1:x2] = 1
            else:
                continue
            x1, y1, x2, y2 = ev.bboxes[local_idx]
            ph, pw = patch.shape[0], patch.shape[1]
            rh, rw = y2 - y1, x2 - x1
            if ph != rh or pw != rw:
                if HAS_CV2:
                    patch = cv2.resize(patch, (rw, rh), interpolation=cv2.INTER_LINEAR)
                else:
                    from PIL import Image
                    patch = np.array(Image.fromarray(patch).resize((rw, rh), Image.Resampling.LANCZOS))
            else:
                patch = patch.copy()
            if self.use_color_matching:
                target_region = frame[y1:y2, x1:x2]
                mask_region = mask[y1:y2, x1:x2]
                if mask_region.shape[:2] == patch.shape[:2]:
                    patch = _color_match_region(patch, target_region, mask_region)
            m = mask[y1:y2, x1:x2]
            if m.ndim == 2:
                m = m[:, :, np.newaxis].astype(np.float32) / 255.0
            else:
                m = m.astype(np.float32) / 255.0
            out[y1:y2, x1:x2] = out[y1:y2, x1:x2] * (1 - m) + patch.astype(np.float32) * m
            if self.debug_draw_border:
                _draw_bbox_border(
                    out, x1, y1, x2, y2,
                    self.debug_border_color,
                    self.debug_border_thickness,
                )
        return np.clip(out, 0, 255).astype(np.uint8)

    def run(
        self,
        frames: List[np.ndarray],
        insertion_events: List[InsertionEvent],
        get_patch_for: callable,
    ) -> List[np.ndarray]:
        """
        get_patch_for(ev_idx, frame_index, ev, frame) -> patch_rgb для этого кадра и события.
        Возвращает список кадров с наложенным контентом.
        """
        result = []
        for fi, frame in enumerate(frames):
            patches = []
            for ev_idx, ev in enumerate(insertion_events):
                if not (ev.start_frame <= fi < ev.end_frame):
                    continue
                patch = get_patch_for(ev_idx, fi, ev, frame)
                if patch is not None:
                    patches.append((ev_idx, patch))
            result.append(self.composite_frame(frame, insertion_events, fi, patches))
        return result
