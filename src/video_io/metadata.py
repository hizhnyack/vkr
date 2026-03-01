"""Структура метаданных видео для сохранения параметров входа и кодирования выхода."""
from __future__ import annotations

from pydantic import BaseModel, Field


class VideoMetadata(BaseModel):
    """Параметры видео: разрешение, битрейт, кодек, FPS, длительность."""

    width: int = Field(..., description="Ширина кадра (px)")
    height: int = Field(..., description="Высота кадра (px)")
    fps: float = Field(..., description="Кадров в секунду")
    duration_sec: float = Field(..., description="Длительность в секундах")
    num_frames: int = Field(..., description="Число кадров")
    codec: str = Field(default="libx264", description="Кодек видео (для выхода)")
    bitrate: str | None = Field(default=None, description="Битрейт, например 2M или 2000k")
    pixel_format: str = Field(default="yuv420p", description="Pixel format для кодирования")
    input_codec: str | None = Field(default=None, description="Кодек входного файла (из ffprobe)")

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)
