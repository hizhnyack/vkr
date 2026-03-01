"""Чтение/запись видео, извлечение метаданных, сохранение параметров выхода."""
from .metadata import VideoMetadata
from .reader import VideoReader
from .writer import VideoWriter

__all__ = ["VideoMetadata", "VideoReader", "VideoWriter"]
