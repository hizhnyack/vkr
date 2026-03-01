"""Тесты планировщика вставки: консистентность масок и траекторий."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from synthesis import InsertionPlanner, InsertionEvent
from scenario_parser import ScenarioEvent, ScenarioObject, EventType, ObjectRole, ZoneType
from scenario_parser.schema import TimeInterval


def test_planner_output_shapes():
    """Планировщик возвращает маски и bbox того же размера кадра."""
    width, height = 320, 240
    fps = 25.0
    num_frames = 100
    planner = InsertionPlanner(width, height, fps, num_frames)
    scenario_events = [
        ScenarioEvent(
            event_type=EventType.PEDESTRIAN_CROSSING,
            objects=[ScenarioObject(role=ObjectRole.PEDESTRIAN, count=1)],
            interval=TimeInterval(start_sec=1, end_sec=4),
            zone=ZoneType.CROSSWALK,
        )
    ]
    events = planner.plan(scenario_events)
    assert len(events) == 1
    ev = events[0]
    assert ev.start_frame >= 0 and ev.end_frame <= num_frames
    if ev.masks:
        for m in ev.masks:
            assert m.shape == (height, width)
    if ev.bboxes:
        for (x1, y1, x2, y2) in ev.bboxes:
            assert 0 <= x1 < x2 <= width
            assert 0 <= y1 < y2 <= height
