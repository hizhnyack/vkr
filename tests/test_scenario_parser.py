"""Тесты парсинга сценария: фиксированные тексты -> ожидаемая структура."""
from __future__ import annotations

import sys
from pathlib import Path

# Корень проекта
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from scenario_parser import parse_task_text, Scenario, ScenarioEvent, EventType, ObjectRole, ZoneType
from scenario_parser.schema import TimeInterval


def test_parse_fallback_without_llm():
    """Без конфига и модели возвращается сценарий-заглушка (одно событие other)."""
    scenario = parse_task_text("Один пешеход переходит дорогу", config=None, llm_model=None, api_client=None)
    assert isinstance(scenario, Scenario)
    assert len(scenario.events) >= 1
    assert scenario.events[0].event_type == EventType.OTHER
    assert scenario.events[0].interval.start_sec >= 0
    assert scenario.events[0].interval.end_sec >= scenario.events[0].interval.start_sec
    assert scenario.raw_text == "Один пешеход переходит дорогу"


def test_time_interval_end_after_start():
    """TimeInterval корректирует end_sec если меньше start_sec."""
    ti = TimeInterval(start_sec=10, end_sec=5)
    assert ti.end_sec >= ti.start_sec, "end_sec должен быть >= start_sec после валидации"


def test_scenario_event_structure():
    """Структура ScenarioEvent и объектов валидна."""
    from scenario_parser import ScenarioObject
    ev = ScenarioEvent(
        event_type=EventType.PEDESTRIAN_CROSSING,
        objects=[ScenarioObject(role=ObjectRole.PEDESTRIAN, count=2)],
        interval=TimeInterval(start_sec=0, end_sec=15),
        zone=ZoneType.CROSSWALK,
    )
    assert ev.event_type == EventType.PEDESTRIAN_CROSSING
    assert len(ev.objects) == 1
    assert ev.objects[0].count == 2
    assert ev.interval.end_sec == 15
