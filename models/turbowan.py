# -*- coding: utf-8 -*-
"""TurboWan 14B 720p: полный GPU при VRAM >= 24 ГБ, иначе CPU offload."""

import os
from typing import Callable, List

from .utils import use_full_gpu

_FULL_GPU_VRAM_GB = 24

metadata = {
    "id": "turbowan_14b_720p",
    "name": "TurboWan 14B 720p",
    "description": "TurboDiffusion, 3–4 шага, требуется turbodiffusion",
    "min_vram_gb": 24,
}


def run(
    prompts: List[str],
    output_dir: str,
    log: Callable[[str], None],
) -> List[str]:
    from .stub import run as run_stub
    try:
        import torch
        from diffusers import WanPipeline
        from diffusers.utils import export_to_video
        model_id = "TurboDiffusion/TurboWan2.1-T2V-14B-720P"
        log("Загрузка TurboWan 14B 720p...")
        pipe = WanPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16)
        if use_full_gpu(_FULL_GPU_VRAM_GB, "TurboWan 14B 720p", log):
            pipe.to("cuda")
        else:
            pipe.enable_model_cpu_offload()
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for i, prompt in enumerate(prompts):
            log("Генерация сегмента %d/%d (TurboWan 14B 720p)..." % (i + 1, len(prompts)))
            out = pipe(
                prompt=prompt,
                num_frames=81,
                height=704,
                width=1280,
                num_inference_steps=4,
                guidance_scale=1.0,
                output_type="np",
            )
            path = os.path.join(output_dir, "seg_%02d.mp4" % i)
            frames = out.frames[0] if hasattr(out, "frames") else out[0]
            export_to_video(frames, path, fps=24)
            paths.append(path)
        return (paths, {"total_steps": len(prompts) * 4})
    except Exception as e:
        log("TurboWan недоступен (%s), используем заглушку." % e)
        return run_stub(prompts, output_dir, log)
