# -*- coding: utf-8 -*-
"""Общие утилиты для моделей: проверка VRAM, выбор режима GPU/offload."""

from typing import Callable, Optional


def get_vram_gb() -> float:
    """Возвращает объём видеопамяти в ГБ (0 если CUDA недоступна)."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def use_full_gpu(
    threshold_gb: float,
    model_name: str,
    log: Callable[[str], None],
) -> bool:
    """
    Очищает кэш CUDA, проверяет объём VRAM. Возвращает True, если VRAM >= threshold_gb
    (рекомендуется загружать пайплайн целиком на GPU). Иначе False — использовать offload.
    Пишет в log сообщение о выборе режима.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            log("%s: CPU offload (CUDA недоступна)" % model_name)
            return False
        torch.cuda.empty_cache()
        vram_gb = get_vram_gb()
        if vram_gb >= threshold_gb:
            log("%s: пайплайн на GPU (VRAM %.1f ГБ >= %.0f ГБ)" % (model_name, vram_gb, threshold_gb))
            return True
        log("%s: CPU offload (VRAM %.1f ГБ < %.0f ГБ)" % (model_name, vram_gb, threshold_gb))
        return False
    except Exception:
        log("%s: CPU offload (ошибка проверки VRAM)" % model_name)
        return False
