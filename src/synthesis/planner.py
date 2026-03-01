"""Планирование вставки: маски, траектории, тайминг по сценарию и сцене."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

# Без opencv — используем numpy для масок; при наличии cv2 — можно эллипсы
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False


@dataclass
class InsertionEvent:
    """Одно событие вставки: интервал кадров, маска или bbox, тип объекта, траектория."""

    start_frame: int
    end_frame: int
    object_type: str  # role из сценария
    trajectory: List[Tuple[int, int, int, int]]  # (frame_idx, x_center, y_center, radius_or_width)
    masks: Optional[List[np.ndarray]] = None  # по кадрам, бинарные (H,W)
    bboxes: Optional[List[Tuple[int, int, int, int]]] = None  # (x1,y1,x2,y2) по кадрам


class InsertionPlanner:
    """По сценарию, описанию сцены и метаданным видео строит список InsertionEvent с масками и траекториями."""

    def __init__(self, width: int, height: int, fps: float, num_frames: int):
        self.width = width
        self.height = height
        self.fps = fps
        self.num_frames = num_frames

    def plan(
        self,
        scenario_events: list,
        scene_descriptions: Optional[list] = None,
    ) -> List[InsertionEvent]:
        """
        scenario_events — список ScenarioEvent (из scenario_parser).
        scene_descriptions — список SceneDescription по кадрам (опционально, для учёта глубины).
        """
        insertion_events = []
        for ev in scenario_events:
            start_sec = ev.interval.start_sec
            end_sec = ev.interval.end_sec
            start_frame = int(start_sec * self.fps)
            end_frame = min(int(end_sec * self.fps), self.num_frames)
            if start_frame >= end_frame:
                start_frame = 0
                end_frame = min(60, self.num_frames)

            # Зона в кадре по zone
            cx, cy = self.width // 2, self.height // 2
            if ev.zone and ev.zone.value == "intersection":
                cx, cy = self.width // 2, int(self.height * 0.55)
            elif ev.zone and ev.zone.value == "crosswalk":
                cx, cy = self.width // 2, int(self.height * 0.7)
            elif ev.zone and ev.zone.value == "side":
                cx, cy = int(self.width * 0.25), self.height // 2

            # Траектория: линейное движение по кадрам (один регион на событие)
            n_frames = end_frame - start_frame
            if n_frames < 1:
                n_frames = 1
            trajectory = []
            bboxes = []
            # Размер области по количеству объектов
            count = sum(getattr(obj, "count", 1) for obj in ev.objects) or 1
            w = min(200, 60 + count * 30)
            h = min(200, 100 + count * 20)
            for i in range(n_frames):
                t = i / max(n_frames - 1, 1)
                x = int(cx + (1 - t) * 40)
                y = int(cy - t * 30)
                trajectory.append((start_frame + i, x, y, max(w, h) // 2))
                x1 = max(0, x - w // 2)
                y1 = max(0, y - h // 2)
                x2 = min(self.width, x + w // 2)
                y2 = min(self.height, y + h // 2)
                bboxes.append((x1, y1, x2, y2))

            masks = self._trajectory_to_masks(trajectory, bboxes)
            insertion_events.append(
                InsertionEvent(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    object_type=getattr(ev.objects[0], "role", "other").value if ev.objects else "other",
                    trajectory=trajectory,
                    masks=masks,
                    bboxes=bboxes,
                )
            )
        return insertion_events

    def _trajectory_to_masks(
        self,
        trajectory: List[Tuple[int, int, int, int]],
        bboxes: List[Tuple[int, int, int, int]],
    ) -> List[np.ndarray]:
        """По траектории и bbox построить бинарные маски по кадрам."""
        masks = []
        for (x1, y1, x2, y2) in bboxes:
            mask = np.zeros((self.height, self.width), dtype=np.uint8)
            if HAS_CV2:
                cv2.rectangle(mask, (x1, y1), (x2, y2), 1, -1)
            else:
                mask[y1:y2, x1:x2] = 1
            masks.append(mask)
        return masks
