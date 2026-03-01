"""Чтение видео по кадрам и извлечение метаданных (decord + ffprobe)."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterator

import numpy as np

from .metadata import VideoMetadata

try:
    from decord import VideoReader as DecordVideoReader
    from decord import cpu
    DECORD_AVAILABLE = True
except ImportError:
    DECORD_AVAILABLE = False


def get_metadata_ffprobe(video_path: Path | str) -> VideoMetadata:
    """Извлечь метаданные через ffprobe."""
    path = str(Path(video_path).resolve())
    cmd = [
        "ffprobe",
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    import json
    data = json.loads(result.stdout)

    width = height = fps = duration_sec = 0
    codec = "libx264"
    bitrate = None
    pixel_format = "yuv420p"
    input_codec = None

    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = int(stream.get("width", 0))
            height = int(stream.get("height", 0))
            input_codec = stream.get("codec_name")
            # FPS from avg_frame_rate "30/1" or "30000/1001"
            rate = stream.get("r_frame_rate") or stream.get("avg_frame_rate", "30/1")
            if "/" in rate:
                a, b = rate.split("/")
                fps = float(a) / float(b) if float(b) else 30.0
            else:
                fps = float(rate)
            break

    fmt = data.get("format", {})
    duration_sec = float(fmt.get("duration", 0))
    if "bit_rate" in fmt:
        bitrate = fmt["bit_rate"]  # bps as string

    if not width or not height:
        raise ValueError(f"Не удалось получить разрешение из {path}")

    num_frames = int(duration_sec * fps) if fps and duration_sec else 0
    if bitrate and bitrate.isdigit():
        bitrate = f"{int(bitrate) // 1000}k"

    return VideoMetadata(
        width=width,
        height=height,
        fps=fps,
        duration_sec=duration_sec,
        num_frames=num_frames,
        codec=codec,
        bitrate=bitrate,
        pixel_format=pixel_format,
        input_codec=input_codec,
    )


class VideoReader:
    """Чтение кадров видео с сохранением FPS; метаданные через ffprobe."""

    def __init__(self, path: Path | str):
        self.path = Path(path).resolve()
        self._metadata: VideoMetadata | None = None
        self._reader = None
        if DECORD_AVAILABLE:
            self._reader = DecordVideoReader(str(self.path), ctx=cpu(0))

    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            self._metadata = get_metadata_ffprobe(self.path)
        return self._metadata

    def __len__(self) -> int:
        if self._reader is not None:
            return len(self._reader)
        return self.metadata.num_frames

    def get_frame(self, frame_index: int) -> np.ndarray:
        """Вернуть кадр как RGB numpy (H, W, 3), uint8."""
        if self._reader is not None:
            frame = self._reader[frame_index]
            if hasattr(frame, "asnumpy"):
                frame = frame.asnumpy()
            # decord returns (H,W,3) RGB
            return np.ascontiguousarray(frame)
        raise RuntimeError("decord не установлен, чтение по кадрам недоступно")

    def iter_frames(self, start: int = 0, end: int | None = None) -> Iterator[np.ndarray]:
        """Итератор по кадрам [start, end)."""
        end = end or len(self)
        for i in range(start, min(end, len(self))):
            yield self.get_frame(i)

    def sample_frames(self, num_samples: int) -> list[tuple[int, np.ndarray]]:
        """Равномерно по времени + первый и последний кадр. Возвращает [(index, frame), ...]."""
        n = len(self)
        if n == 0:
            return []
        indices = [0]
        if num_samples > 2:
            step = (n - 1) / (num_samples - 1)
            for i in range(1, num_samples - 1):
                indices.append(int(round(i * step)))
        if n > 1:
            indices.append(n - 1)
        return [(i, self.get_frame(i)) for i in indices]
