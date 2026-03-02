"""Структура описания сцены: время суток, погода, освещение."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class SceneDescription(BaseModel):
    """Описание сцены по кадру/видео. Карта глубины передаётся отдельно (не в JSON)."""

    time_of_day: str = Field(default="day", description="day, night, dawn, dusk")
    weather: str = Field(default="clear", description="clear, rain, fog, snow, overcast")
    lighting: str = Field(default="natural", description="natural, artificial, mixed")
    lighting_direction: Optional[str] = Field(default=None)
    frame_index: Optional[int] = Field(default=None)
    depth_map: Optional[Any] = Field(default=None, exclude=True, description="Карта глубины (numpy), не сериализуется в JSON")

    class Config:
        arbitrary_types_allowed = True

    def set_depth_map(self, depth_map: Any) -> None:
        self.depth_map = depth_map

    def get_depth_map(self) -> Optional[Any]:
        return getattr(self, "depth_map", None)
