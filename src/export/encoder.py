"""Кодирование итоговых кадров в видео с параметрами исходного."""
from __future__ import annotations

from pathlib import Path
from typing import Iterator, List

import numpy as np

from video_io.metadata import VideoMetadata
from video_io.writer import VideoWriter


def export_video(
    frames: List[np.ndarray],
    output_path: Path | str,
    metadata: VideoMetadata,
) -> None:
    """
    Записать кадры в видео через ffmpeg с теми же параметрами, что и вход.
    frames — список RGB (H,W,3) uint8; размер должен совпадать с metadata.width x metadata.height.
    """
    output_path = Path(output_path)
    writer = VideoWriter(output_path, metadata)
    def gen():
        for f in frames:
            yield f
    writer.write_frames(gen())
