"""Парсинг текста задачи в структурированный сценарий через LLM (local или API)."""
from __future__ import annotations

import json
import re
import logging
from typing import Optional

from .schema import Scenario, ScenarioEvent, ScenarioObject, TimeInterval, ObjectRole, EventType, ZoneType

logger = logging.getLogger(__name__)

SCENARIO_JSON_PROMPT = """Ты — ассистент, который извлекает структурированное описание дорожного сценария из текста пользователя.

По описанию задачи сформируй JSON со следующей структурой (строго придерживайся имён полей):
{
  "events": [
    {
      "event_type": "<один из: accident, pedestrian_crossing, pedestrian_collision, emergency_vehicle, fire, infrastructure_failure, weather_hazard, debris_on_road, unusual_behavior, crowd, other>",
      "objects": [
        {
          "role": "<один из: car, truck, pedestrian, child, bicycle, special_vehicle, fire, smoke, debris, suspicious_object, animal, other>",
          "count": 1,
          "size": "medium или null",
          "speed": "medium или null",
          "description": "краткое уточнение или null"
        }
      ],
      "interval": { "start_sec": 0, "end_sec": 10 },
      "zone": "<один из: intersection, lane, crosswalk, center, side, full_frame, unspecified>",
      "description": "краткое описание события или null"
    }
  ]
}

Правила:
- Если время не указано, используй start_sec: 0, end_sec: 30.
- objects — массив; role и count обязательны.
- Ответь ТОЛЬКО валидным JSON, без markdown и пояснений до или после.
"""


def _extract_json_from_response(text: str) -> str:
    """Извлечь JSON из ответа LLM (убрать markdown, обрезки)."""
    text = text.strip()
    # Убрать markdown code block
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        return text[start:end].strip()
    if "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        return text[start:end].strip()
    # Найти первый { и последний }
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        return text[first : last + 1]
    return text


def _parse_llm_json_to_scenario(data: dict, raw_text: Optional[str] = None) -> Scenario:
    """Преобразовать словарь из LLM в Scenario (Pydantic)."""
    events = []
    for ev in data.get("events", []):
        interval = ev.get("interval", {})
        ti = TimeInterval(
            start_sec=float(interval.get("start_sec", 0)),
            end_sec=float(interval.get("end_sec", 30)),
        )
        objs = []
        for o in ev.get("objects", []):
            role_str = (o.get("role") or "other").lower()
            try:
                role = ObjectRole(role_str)
            except ValueError:
                role = ObjectRole.OTHER
            objs.append(
                ScenarioObject(
                    role=role,
                    count=int(o.get("count", 1)),
                    size=o.get("size"),
                    speed=o.get("speed"),
                    description=o.get("description"),
                )
            )
        event_type_str = (ev.get("event_type") or "other").lower()
        try:
            event_type = EventType(event_type_str)
        except ValueError:
            event_type = EventType.OTHER
        zone_str = (ev.get("zone") or "unspecified").lower()
        try:
            zone = ZoneType(zone_str)
        except ValueError:
            zone = ZoneType.UNSPECIFIED
        events.append(
            ScenarioEvent(
                event_type=event_type,
                objects=objs,
                interval=ti,
                zone=zone,
                description=ev.get("description"),
            )
        )
    return Scenario(events=events, raw_text=raw_text)


def parse_task_text(
    task_text: str,
    config: Optional[object] = None,
    llm_model = None,
    api_client: Optional[object] = None,
) -> Scenario:
    """
    Преобразовать текст задачи в структурированный сценарий.
    config — объект с атрибутами llm.mode, llm.model_name, llm.load_in_4bit, llm.api_key, llm.api_base.
    llm_model — предзагруженная модель (для mode=local).
    api_client — клиент API (для mode=openai/anthropic).
    """
    prompt = SCENARIO_JSON_PROMPT + "\n\nТекст задачи пользователя:\n" + task_text.strip()

    mode = getattr(getattr(config, "llm", None), "mode", "local") if config else "local"
    mode = mode or "local"

    # Резерв: без конфига и без модели/API — минимальный сценарий по тексту (для тестов)
    if config is None and llm_model is None and api_client is None:
        return _fallback_scenario_from_text(task_text)

    if mode == "openai" and api_client is not None:
        raw = _call_openai(api_client, prompt, config)
    elif mode == "anthropic" and api_client is not None:
        raw = _call_anthropic(api_client, prompt, config)
    else:
        raw = _call_local_llm(llm_model, prompt, config)

    json_str = _extract_json_from_response(raw or "")
    if not json_str or not json_str.strip():
        logger.warning("LLM вернул пустой ответ, используем запасной сценарий")
        return _fallback_scenario_from_text(task_text)

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.warning("LLM вернул невалидный JSON, пробуем исправить: %s", e)
        json_str = re.sub(r",\s*}", "}", json_str)
        json_str = re.sub(r",\s*]", "]", json_str)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning("Исправление не помогло, используем запасной сценарий")
            return _fallback_scenario_from_text(task_text)

    return _parse_llm_json_to_scenario(data, raw_text=task_text.strip())


def _call_local_llm(model, prompt: str, config: Optional[object]) -> str:
    """Вызов локальной модели (transformers)."""
    if model is None:
        # Ленивая загрузка при первом вызове
        model = _load_local_llm(config)
    import torch
    from transformers import AutoTokenizer

    tokenizer = getattr(model, "tokenizer", None) or AutoTokenizer.from_pretrained(
        getattr(getattr(config, "llm", None), "model_name", "Qwen/Qwen2.5-3B-Instruct")
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()


def _load_local_llm(config: Optional[object]):
    """Загрузить локальную LLM (4-bit при необходимости; при нехватке VRAM — без квантизации)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    cfg = getattr(config, "llm", None) or object()
    model_name = getattr(cfg, "model_name", "Qwen/Qwen2.5-3B-Instruct")
    load_4bit = getattr(cfg, "load_in_4bit", True)
    device = getattr(config, "device", "cuda")

    quantization = None
    if load_4bit:
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_quant_type="nf4",
        )
    device_map = "auto" if device == "cuda" else None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization,
            device_map=device_map,
            trust_remote_code=True,
        )
    except ValueError as e:
        if "CPU or the disk" in str(e) or "quantized model" in str(e):
            logger.warning(
                "Недостаточно VRAM для 4-bit квантизации, загружаем модель без квантизации (float16): %s",
                e,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=None,
                device_map=device_map,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
            )
        else:
            raise
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if device == "cpu" and quantization is None:
        model = model.to("cpu")
    model.tokenizer = tokenizer
    return model


def _fallback_scenario_from_text(task_text: str) -> Scenario:
    """Минимальный сценарий без LLM: одно событие other, интервал 0–30 сек."""
    return Scenario(
        events=[
            ScenarioEvent(
                event_type=EventType.OTHER,
                objects=[ScenarioObject(role=ObjectRole.OTHER, count=1, description=task_text[:200])],
                interval=TimeInterval(start_sec=0, end_sec=30),
                zone=ZoneType.UNSPECIFIED,
                description=task_text[:500],
            )
        ],
        raw_text=task_text.strip(),
    )


def _call_openai(client, prompt: str, config: Optional[object]) -> str:
    """Вызов OpenAI API."""
    resp = client.chat.completions.create(
        model=getattr(getattr(config, "llm", None), "model_name", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    return (resp.choices[0].message.content or "").strip()


def _fallback_scenario_from_text(task_text: str) -> Scenario:
    """Минимальный сценарий без LLM: одно событие other, интервал 0–30 сек."""
    return Scenario(
        events=[
            ScenarioEvent(
                event_type=EventType.OTHER,
                objects=[ScenarioObject(role=ObjectRole.OTHER, count=1, description=task_text[:200])],
                interval=TimeInterval(start_sec=0, end_sec=30),
                zone=ZoneType.UNSPECIFIED,
                description=task_text[:500],
            )
        ],
        raw_text=task_text.strip(),
    )


def _call_anthropic(client, prompt: str, config: Optional[object]) -> str:
    """Вызов Anthropic API."""
    resp = client.messages.create(
        model=getattr(getattr(config, "llm", None), "model_name", "claude-3-haiku-20240307"),
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return (resp.content[0].text if resp.content else "").strip()
