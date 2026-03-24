# -*- coding: utf-8 -*-
"""Wan 2.1 T2V-1.3B: полный GPU при VRAM >= 20 ГБ, иначе CPU offload."""

import os
from typing import Callable, List

from .utils import use_full_gpu

_FULL_GPU_VRAM_GB = 20

metadata = {
    "id": "wan21_1_3b",
    "name": "Wan 2.1 1.3B",
    "description": "480p, ~5 сек",
    "min_vram_gb": None,
}


def run(
    prompts: List[str],
    output_dir: str,
    log: Callable[[str], None],
) -> List[str]:
    from .stub import run as run_stub
    try:
        from diffusers import WanPipeline
        from diffusers.utils import export_to_video
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
        vae = None
        try:
            from diffusers import AutoencoderKLWan
            vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        except Exception:
            pass
        text_encoder = None
        tokenizer = None
        try:
            from transformers import UMT5ForConditionalGeneration, AutoTokenizer
            log("Загрузка текстового энкодера UMT5-XXL (google/umt5-xxl)...")
            umt5 = UMT5ForConditionalGeneration.from_pretrained("google/umt5-xxl", torch_dtype=torch.bfloat16)
            text_encoder = umt5.encoder
            del umt5
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            tokenizer = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        except Exception as e:
            log("Текстовый энкодер из google/umt5-xxl не загружен (%s), используем из репо Wan." % e)
        kwargs = {"vae": vae, "torch_dtype": torch.bfloat16}
        if text_encoder is not None:
            kwargs["text_encoder"] = text_encoder
        if tokenizer is not None:
            kwargs["tokenizer"] = tokenizer
        pipe = WanPipeline.from_pretrained(model_id, **kwargs)
        if text_encoder is not None:
            pipe.text_encoder = text_encoder
        if use_full_gpu(_FULL_GPU_VRAM_GB, "Wan 2.1 1.3B", log):
            pipe.to("cuda")
        else:
            pipe.enable_model_cpu_offload()
        os.makedirs(output_dir, exist_ok=True)
        paths = []
        for i, prompt in enumerate(prompts):
            log("Генерация сегмента %d/%d (Wan 2.1 1.3B)..." % (i + 1, len(prompts)))
            out = pipe(
                prompt=prompt,
                num_frames=81,
                height=480,
                width=832,
                guidance_scale=5.0,
                num_inference_steps=50,
                output_type="np",
            )
            path = os.path.join(output_dir, "seg_%02d.mp4" % i)
            frames = out.frames[0] if hasattr(out, "frames") else out[0]
            export_to_video(frames, path, fps=16)
            paths.append(path)
        return (paths, {"total_steps": len(prompts) * 50})
    except ImportError as e:
        log("Wan 2.1: модуль не установлен (%s), используем заглушку." % e)
        return run_stub(prompts, output_dir, log)
    except Exception as e:
        log("Ошибка Wan 2.1: %s" % e)
        raise RuntimeError("Wan 2.1: %s" % e)
