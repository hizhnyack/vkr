"""Тесты видео I/O: метаданные, консистентность параметров при записи."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from video_io import VideoMetadata, VideoWriter


def test_metadata_resolution():
    """VideoMetadata хранит разрешение и fps."""
    meta = VideoMetadata(
        width=1920,
        height=1080,
        fps=25.0,
        duration_sec=10.0,
        num_frames=250,
    )
    assert meta.resolution == (1920, 1080)
    assert meta.num_frames == 250


def test_write_frames_matches_metadata():
    """Запись кадров с заданными метаданными создаёт файл (проверка наличия)."""
    import shutil
    if not shutil.which("ffmpeg"):
        return  # пропуск без pytest
    meta = VideoMetadata(
        width=64,
        height=64,
        fps=10.0,
        duration_sec=0.5,
        num_frames=5,
    )
    frames = [np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8) for _ in range(5)]
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        out_path = f.name
    try:
        writer = VideoWriter(out_path, meta)
        writer.write_frames_list(frames)
        assert Path(out_path).exists()
        assert Path(out_path).stat().st_size > 0
    finally:
        Path(out_path).unlink(missing_ok=True)
