# -*- coding: utf-8 -*-
"""Allegro-T2V-40x360P: полный GPU при VRAM >= 12 ГБ, иначе CPU offload."""

import os
from typing import Callable, List

from .utils import use_full_gpu

_FULL_GPU_VRAM_GB = 12

metadata = {
    "id": "allegro_360",
    "name": "Allegro 360P",
    "description": "Лёгкая, полный GPU при достаточной VRAM",
    "min_vram_gb": None,
}


def run(
    prompts: List[str],
    output_dir: str,
    log: Callable[[str], None],
) -> List[str]:
    try:
        from diffusers import AllegroPipeline
        from diffusers.utils import export_to_video
        import torch
        pipe = AllegroPipeline.from_pretrained("rhymes-ai/Allegro-T2V-40x360P", torch_dtype=torch.float16)
        if use_full_gpu(_FULL_GPU_VRAM_GB, "Allegro 360P", log):
            pipe.to("cuda")
        else:
            pipe.enable_sequential_cpu_offload()
        pipe.vae.enable_tiling()
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        num_steps = 28  # default inference steps per segment
        for i, prompt in enumerate(prompts):
            log("Генерация сегмента %d/%d (Allegro 360P)..." % (i + 1, len(prompts)))
            out = pipe(prompt=prompt, num_frames=40, height=368, width=640)
            path = os.path.join(output_dir, "seg_%02d.mp4" % i)
            frames = out.frames[0] if hasattr(out, "frames") else out
            export_to_video(frames, path, fps=15)
            paths.append(path)
        return (paths, {"total_steps": len(prompts) * num_steps})
    except Exception as e:
        log("Ошибка Allegro 360P: %s" % e)
        raise RuntimeError("Allegro 360P: %s" % e)
