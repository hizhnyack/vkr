"""Pydantic-схема сценария: тип события, объекты, временные интервалы, зона."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class ObjectRole(str, Enum):
    """Роль объекта в сценарии."""
    CAR = "car"
    TRUCK = "truck"
    PEDESTRIAN = "pedestrian"
    CHILD = "child"
    BICYCLE = "bicycle"
    SPECIAL_VEHICLE = "special_vehicle"  # скорая, полиция, МЧС
    FIRE = "fire"
    SMOKE = "smoke"
    DEBRIS = "debris"  # груз, обломки
    SUSPICIOUS_OBJECT = "suspicious_object"
    ANIMAL = "animal"
    OTHER = "other"


class EventType(str, Enum):
    """Тип события."""
    ACCIDENT = "accident"
    PEDESTRIAN_CROSSING = "pedestrian_crossing"
    PEDESTRIAN_COLLISION = "pedestrian_collision"
    EMERGENCY_VEHICLE = "emergency_vehicle"
    FIRE = "fire"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"
    WEATHER_HAZARD = "weather_hazard"
    DEBRIS_ON_ROAD = "debris_on_road"
    UNUSUAL_BEHAVIOR = "unusual_behavior"
    CROWD = "crowd"
    OTHER = "other"


class ZoneType(str, Enum):
    """Зона действия в кадре."""
    INTERSECTION = "intersection"
    LANE = "lane"
    CROSSWALK = "crosswalk"
    CENTER = "center"
    SIDE = "side"
    FULL_FRAME = "full_frame"
    UNSPECIFIED = "unspecified"


class ScenarioObject(BaseModel):
    """Один объект в сценарии."""
    role: ObjectRole
    count: int = Field(1, ge=1)
    size: Optional[str] = None   # small / medium / large
    speed: Optional[str] = None  # slow / medium / fast
    description: Optional[str] = None


class TimeInterval(BaseModel):
    """Временной интервал в секундах."""
    start_sec: float = Field(..., ge=0)
    end_sec: float = Field(..., ge=0)

    @model_validator(mode="after")
    def end_after_start(self):
        if self.end_sec < self.start_sec:
            object.__setattr__(self, "end_sec", self.start_sec)
        return self


class ScenarioEvent(BaseModel):
    """Одно событие: тип, объекты, интервал, зона."""
    event_type: EventType
    objects: list[ScenarioObject] = Field(default_factory=list)
    interval: TimeInterval
    zone: ZoneType = ZoneType.UNSPECIFIED
    description: Optional[str] = None


class Scenario(BaseModel):
    """Полный сценарий: список событий, извлечённых из текста задачи."""
    events: list[ScenarioEvent] = Field(default_factory=list)
    raw_text: Optional[str] = None
