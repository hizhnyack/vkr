"""Опциональная разметка: детекция по выходному видео, экспорт в JSON."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


def export_annotations_if_requested(
    video_path: Path | str,
    output_json_path: Optional[Path | str] = None,
    model_name: str = "yolov8n.pt",
    every_n_frames: int = 1,
) -> Optional[dict]:
    """
    Прогон детектора по видео; экспорт в JSON: по кадрам (или по every_n_frames) — список bbox с классом и временем.
    Если output_json_path задан — записать туда JSON. Возвращает словарь с разметкой или None при ошибке.
    """
    if not YOLO_AVAILABLE:
        logger.warning("ultralytics не установлен, разметка пропущена")
        return None
    video_path = Path(video_path)
    try:
        model = YOLO(model_name)
        results = model(str(video_path), stream=True, verbose=False)
        annotations = {"frames": [], "model": model_name}
        frame_idx = 0
        for r in results:
            if every_n_frames > 0 and frame_idx % every_n_frames != 0:
                frame_idx += 1
                continue
            boxes = []
            if r.boxes is not None:
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    cls_id = int(box.cls[0].item())
                    name = model.names.get(cls_id, "unknown")
                    boxes.append({"bbox": xyxy, "class": name, "class_id": cls_id})
            annotations["frames"].append({
                "frame_index": frame_idx,
                "time_sec": frame_idx / 25.0,
                "detections": boxes,
            })
            frame_idx += 1
        if output_json_path:
            Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
            with open(output_json_path, "w", encoding="utf-8") as f:
                json.dump(annotations, f, ensure_ascii=False, indent=2)
        return annotations
    except Exception as e:
        logger.warning("Экспорт разметки не удался: %s", e)
        return None
