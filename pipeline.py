# -*- coding: utf-8 -*-
"""
Пайплайн генерации видео по текстовому промпту:
перевод RU→EN, разбиение на сегменты, T2V, сшивка.
"""

import json
import os
import re
import time
from typing import Callable, List, Optional

from models import MODEL_RUNNERS, run_stub

_DEBUG_LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cursor")
_DEBUG_LOG_PATH = os.path.join(_DEBUG_LOG_DIR, "debug.log")


def _log_fallback_error(model_id: str, exc: BaseException) -> None:
    """Пишет в .cursor/debug.log ошибку, из-за которой включился fallback (заглушка)."""
    try:
        os.makedirs(_DEBUG_LOG_DIR, exist_ok=True)
        payload = {
            "message": "model_fallback",
            "model_id": model_id,
            "exception_type": type(exc).__name__,
            "exception_msg": str(exc),
            "timestamp": int(time.time() * 1000),
        }
        with open(_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def _noop_log(msg: str) -> None:
    pass


def translate_ru_en(text: str, log: Callable[[str], None] = _noop_log) -> str:
    """Перевод русского текста на английский (Helsinki-NLP opus-mt-ru-en)."""
    text = (text or "").strip()
    if not text:
        log("Перевод: пустой ввод, используем как есть.")
        return text
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        import torch
        model_name = "Helsinki-NLP/opus-mt-ru-en"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            out = model.generate(**inputs, max_length=512)
        en = tokenizer.decode(out[0], skip_special_tokens=True)
        log("Перевод: " + en[:200] + ("..." if len(en) > 200 else ""))
        return en
    except Exception as e:
        log("Перевод (ошибка, используем исходный текст): " + str(e))
        return text


def split_into_segments(en_text: str, num_segments: int = 3, log: Callable[[str], None] = _noop_log) -> List[str]:
    """Разбивает текст на 2–3 подпромпта для сегментов по 4–6 сек (цель 10–15 сек)."""
    en_text = (en_text or "").strip()
    if not en_text:
        return ["A short video clip."] * max(1, num_segments)
    sentences = re.split(r'(?<=[.!?])\s+', en_text)
    if not sentences:
        segments = [en_text] if en_text else ["A short video clip."]
    else:
        n = min(num_segments, len(sentences))
        per = max(1, len(sentences) // n)
        segments = []
        for i in range(n):
            start = i * per
            end = len(sentences) if i == n - 1 else (i + 1) * per
            seg = " ".join(sentences[start:end]).strip()
            if seg:
                segments.append(seg)
        if not segments:
            segments = [en_text]
    for i, s in enumerate(segments):
        log("Сегмент %d: %s" % (i + 1, s[:100] + ("..." if len(s) > 100 else "")))
    return segments


def _get_video_info(video_path: str) -> dict:
    """Возвращает num_frames, video_width, video_height из файла видео (imageio)."""
    out = {"num_frames": 0, "video_width": 0, "video_height": 0}
    try:
        import imageio
        r = imageio.get_reader(video_path)
        meta = r.get_meta_data()
        size = meta.get("size", (0, 0))
        if isinstance(size, (list, tuple)) and len(size) >= 2:
            out["video_width"] = int(size[0])
            out["video_height"] = int(size[1])
        n = 0
        for _ in r:
            n += 1
        r.close()
        out["num_frames"] = n
    except Exception:
        pass
    return out


def stitch_videos(
    video_paths: List[str],
    output_path: str,
    log: Callable[[str], None] = _noop_log,
) -> None:
    """Склеивает несколько видео в один файл (imageio или ffmpeg)."""
    if not video_paths:
        raise ValueError("Нет файлов для сшивки")
    for p in video_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError("Файл не найден: %s" % p)
    try:
        import imageio
        reader = imageio.get_reader(video_paths[0])
        meta = reader.get_meta_data()
        fps = meta.get("fps") or 8
        reader.close()
        writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=6)
        for path in video_paths:
            r = imageio.get_reader(path)
            for frame in r:
                writer.append_data(frame)
            r.close()
        writer.close()
        log("Сшивка завершена: %s" % output_path)
    except Exception as e:
        log("Сшивка (imageio): %s" % e)
        import subprocess
        list_file = output_path + ".list.txt"
        with open(list_file, "w") as f:
            for p in video_paths:
                f.write("file '%s'\n" % os.path.abspath(p))
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_file,
            "-c", "copy", output_path
        ], check=True)
        try:
            os.remove(list_file)
        except OSError:
            pass
        log("Сшивка завершена (ffmpeg): %s" % output_path)


def run_pipeline(
    prompt_ru: str,
    model_id: str,
    output_dir: str,
    task_id: str,
    set_status: Callable[[str], None],
    set_progress: Callable[[int], None],
    append_log: Callable[[str], None],
    set_output_path: Callable[[str], None],
    set_meta: Optional[Callable[[int, str], None]] = None,
) -> None:
    """Полный пайплайн: перевод, сегменты, T2V, сшивка."""
    try:
        set_status("translating")
        set_progress(5)
        en = translate_ru_en(prompt_ru, append_log)
        segments = split_into_segments(en, num_segments=3, log=append_log)
        set_status("generating")
        runner, use_stub_on_fail = MODEL_RUNNERS.get(model_id, (run_stub, False))
        seg_dir = os.path.join(output_dir, task_id, "segments")
        os.makedirs(seg_dir, exist_ok=True)
        video_paths = []
        runner_stats = {}
        t_gen_start = time.time()
        try:
            result = runner(segments, seg_dir, append_log)
            if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[0], list):
                video_paths, runner_stats = result[0], result[1]
            else:
                video_paths = result if isinstance(result, list) else list(result)
        except Exception as e:
            if use_stub_on_fail:
                _log_fallback_error(model_id, e)
                append_log("Модель недоступна, используем заглушку: %s" % e)
                result = run_stub(segments, seg_dir, append_log)
                if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[0], list):
                    video_paths, runner_stats = result[0], result[1]
                else:
                    video_paths = result if isinstance(result, list) else list(result)
            else:
                raise
        gen_duration_sec = time.time() - t_gen_start
        total_steps = runner_stats.get("total_steps") or 0
        avg_s_per_it = round(gen_duration_sec / total_steps, 3) if total_steps else None
        n = len(segments)
        for i in range(n):
            set_progress(10 + int(70 * (i + 1) / n))
        set_status("stitching")
        set_progress(85)
        out_path = os.path.join(output_dir, task_id, "output.mp4")
        stitch_videos(video_paths, out_path, log=append_log)
        set_progress(100)
        set_output_path(out_path)
        video_info = _get_video_info(out_path)
        if set_meta:
            set_meta(
                n,
                (en[:500] if en else ""),
                avg_s_per_it=avg_s_per_it,
                num_frames=video_info["num_frames"],
                video_width=video_info["video_width"],
                video_height=video_info["video_height"],
            )
        set_status("done")
    except Exception as e:
        append_log("Ошибка: %s" % e)
        set_status("error")
