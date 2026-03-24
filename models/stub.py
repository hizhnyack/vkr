# -*- coding: utf-8 -*-
"""Заглушка: чёрные кадры. Только для fallback, в меню не показывается."""

import os
from typing import Callable, List


def _stub_generate_segment(
    prompt: str,
    output_path: str,
    fps: int = 8,
    duration_sec: float = 5.0,
    log: Callable[[str], None] = lambda _: None,
) -> None:
    import numpy as np
    import imageio
    n_frames = max(1, int(fps * duration_sec))
    h, w = 368, 640
    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=5)
    for _ in range(n_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        writer.append_data(frame)
    writer.close()
    log("Записано тестовое видео (чёрные кадры-заглушка): %d кадров, %s" % (n_frames, output_path))


def run(
    prompts: List[str],
    output_dir: str,
    log: Callable[[str], None],
):
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for i, prompt in enumerate(prompts):
        path = os.path.join(output_dir, "seg_%02d.mp4" % i)
        _stub_generate_segment(prompt, path, fps=8, duration_sec=4.0, log=log)
        paths.append(path)
    return (paths, {})
