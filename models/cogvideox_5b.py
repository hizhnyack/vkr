# -*- coding: utf-8 -*-
"""CogVideoX-5b: полный GPU при VRAM >= 20 ГБ, иначе CPU offload."""

import os
from typing import Callable, List

from .utils import use_full_gpu

_FULL_GPU_VRAM_GB = 20

metadata = {
    "id": "cogvideox_5b",
    "name": "CogVideoX 5B",
    "description": "720p, полный GPU при 20+ ГБ VRAM",
    "min_vram_gb": 20,
}


def run(
    prompts: List[str],
    output_dir: str,
    log: Callable[[str], None],
) -> List[str]:
    from .stub import run as run_stub
    try:
        from diffusers import CogVideoXPipeline
        from diffusers.utils import export_to_video
        import torch
        model_id = "THUDM/CogVideoX-5b"
        pipe = CogVideoXPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        if use_full_gpu(_FULL_GPU_VRAM_GB, "CogVideoX 5B", log):
            pipe.to("cuda")
        else:
            pipe.enable_model_cpu_offload()
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        num_frames = 49
        for i, prompt in enumerate(prompts):
            log("Генерация сегмента %d/%d (CogVideoX 5B)..." % (i + 1, len(prompts)))
            out = pipe(
                prompt=prompt,
                num_frames=num_frames,
                num_inference_steps=50,
            )
            path = os.path.join(output_dir, "seg_%02d.mp4" % i)
            frames = out.frames[0] if hasattr(out, "frames") else out[0]
            export_to_video(frames, path, fps=8)
            paths.append(path)
        return (paths, {"total_steps": len(prompts) * 50})
    except ImportError as e:
        log("CogVideoX 5B: модуль не установлен (%s), используем заглушку." % e)
        return run_stub(prompts, output_dir, log)
    except Exception as e:
        log("Ошибка CogVideoX 5B: %s" % e)
        raise RuntimeError("CogVideoX 5B: %s" % e)
