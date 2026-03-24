# -*- coding: utf-8 -*-
"""Регистр моделей: MODEL_RUNNERS и MODELS_META (без stub для меню)."""

from .stub import run as run_stub

_MODULES = [
    ("allegro_360", True),
    ("allegro_720", True),
    ("wan21_1_3b", True),
    ("wan22_5b", True),
]

MODEL_RUNNERS = {"stub": (run_stub, False)}
MODELS_META = []

for _name, _use_stub_on_fail in _MODULES:
    try:
        mod = __import__("models." + _name, fromlist=["run", "metadata"])
        MODEL_RUNNERS[mod.metadata["id"]] = (mod.run, _use_stub_on_fail)
        MODELS_META.append(mod.metadata.copy())
    except Exception:
        pass

# Опциональные тяжёлые модели (могут отсутствовать)
try:
    from . import wan22_a14b
    MODEL_RUNNERS[wan22_a14b.metadata["id"]] = (wan22_a14b.run, True)
    MODELS_META.append(wan22_a14b.metadata.copy())
except Exception:
    pass

try:
    from . import cogvideox_5b
    MODEL_RUNNERS[cogvideox_5b.metadata["id"]] = (cogvideox_5b.run, True)
    MODELS_META.append(cogvideox_5b.metadata.copy())
except Exception:
    pass

try:
    from . import turbowan
    MODEL_RUNNERS[turbowan.metadata["id"]] = (turbowan.run, True)
    MODELS_META.append(turbowan.metadata.copy())
except Exception:
    pass

__all__ = ["MODEL_RUNNERS", "MODELS_META", "run_stub"]
