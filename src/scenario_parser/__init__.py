"""Парсинг текстовой задачи в структурированный сценарий (LLM + Pydantic)."""
from .schema import Scenario, ScenarioObject, ScenarioEvent, ObjectRole, EventType, ZoneType
from .parser import parse_task_text

__all__ = [
    "Scenario",
    "ScenarioObject",
    "ScenarioEvent",
    "ObjectRole",
    "EventType",
    "ZoneType",
    "parse_task_text",
]
