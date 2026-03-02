"""Загрузка и валидация конфигурации (YAML + dataclass)."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    mode: str = "local"
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    load_in_4bit: bool = True
    api_key: Optional[str] = None
    api_base: Optional[str] = None


class VLMConfig(BaseModel):
    model_name: str = "Salesforce/blip2-opt-2.7b"
    load_in_4bit: bool = True


class DepthConfig(BaseModel):
    model_name: str = "Intel/dpt-hybrid-midas"


class DiffusionConfig(BaseModel):
    model_id: str = "runwayml/stable-diffusion-v1-5"
    controlnet_id: str = "lllyasviel/control_v11f1p_sd15_depth"
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    max_resolution: int = 512


class InpaintingConfig(BaseModel):
    enabled: bool = False
    model_path: Optional[str] = None


class MemoryConfig(BaseModel):
    max_vram_gb: int = 8
    clear_cache_between_models: bool = True


class Config(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    vlm: VLMConfig = Field(default_factory=VLMConfig)
    depth: DepthConfig = Field(default_factory=DepthConfig)
    diffusion: DiffusionConfig = Field(default_factory=DiffusionConfig)
    inpainting: InpaintingConfig = Field(default_factory=InpaintingConfig)
    device: str = "cuda"
    memory: MemoryConfig = Field(default_factory=MemoryConfig)

    @classmethod
    def from_yaml(cls, path: Path | str) -> "Config":
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def to_yaml(self, path: Path | str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.model_dump(), f, allow_unicode=True, default_flow_style=False)
