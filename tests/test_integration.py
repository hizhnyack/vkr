"""Интеграционный тест: сценарий -> планировщик -> композитинг (без реального видео)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from scenario_parser import parse_task_text, ScenarioEvent, ScenarioObject, EventType, ObjectRole
from scenario_parser.schema import TimeInterval, ZoneType
from synthesis import InsertionPlanner, CompositePipeline


def test_pipeline_flow_without_llm():
    """Полный проход: текст -> сценарий (fallback) -> план вставки -> композитинг по синтетическим кадрам."""
    scenario = parse_task_text("Один пешеход переходит дорогу", config=None)
    assert len(scenario.events) >= 1

    width, height, fps, num_frames = 320, 240, 25.0, 50
    planner = InsertionPlanner(width, height, fps, num_frames)
    insertion_events = planner.plan(scenario.events, [])
    assert len(insertion_events) >= 1

    frames = [np.random.randint(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(num_frames)]
    composite = CompositePipeline(use_color_matching=False)

    def get_patch_for(ev_idx, frame_index, ev, frame):
        if ev.bboxes and 0 <= frame_index - ev.start_frame < len(ev.bboxes):
            x1, y1, x2, y2 = ev.bboxes[frame_index - ev.start_frame]
            return frame[y1:y2, x1:x2].copy()
        return None

    out_frames = composite.run(frames, insertion_events, get_patch_for)
    assert len(out_frames) == num_frames
    assert out_frames[0].shape == (height, width, 3)
