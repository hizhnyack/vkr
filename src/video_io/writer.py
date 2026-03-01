"""Запись видео через ffmpeg с параметрами, совпадающими с входом."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np

from .metadata import VideoMetadata


class VideoWriter:
    """Запись кадров в файл с заданными метаданными (разрешение, битрейт, кодек, FPS)."""

    def __init__(self, output_path: Path | str, metadata: VideoMetadata):
        self.output_path = Path(output_path).resolve()
        self.metadata = metadata
        self._process: subprocess.Popen | None = None
        self._pipe_path: str | None = None

    def write_frames(self, frames: Iterator[np.ndarray]) -> None:
        """Принять итератор кадров RGB (H,W,3) uint8 и записать через ffmpeg."""
        # Пайп: stdin -> ffmpeg
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{self.metadata.width}x{self.metadata.height}",
            "-pix_fmt", "rgb24",
            "-r", str(self.metadata.fps),
            "-i", "-",
            "-c:v", self.metadata.codec,
            "-pix_fmt", self.metadata.pixel_format,
            "-r", str(self.metadata.fps),
        ]
        if self.metadata.bitrate:
            cmd.extend(["-b:v", self.metadata.bitrate])
        cmd.append(str(self.output_path))

        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        try:
            for frame in frames:
                if frame.shape[:2] != (self.metadata.height, self.metadata.width):
                    frame = _resize_frame(frame, self.metadata.width, self.metadata.height)
                proc.stdin.write(frame.astype(np.uint8).tobytes())
        except Exception:
            proc.kill()
            proc.wait()
            raise
        finally:
            if proc.stdin is not None and not proc.stdin.closed:
                proc.stdin.close()
        proc.wait()
        stderr = proc.stderr.read() if proc.stderr else b""
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')}")

    def write_frames_list(self, frames: list[np.ndarray]) -> None:
        """Записать список кадров."""
        def gen():
            for f in frames:
                yield f
        self.write_frames(gen())


def _resize_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    try:
        import cv2
        return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image
        img = Image.fromarray(frame)
        img = img.resize((width, height), Image.Resampling.LANCZOS)
        return np.array(img)
