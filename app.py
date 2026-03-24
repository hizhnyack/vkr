# -*- coding: utf-8 -*-
"""
Веб-приложение генерации видео по текстовому промпту.
Flask: маршруты, раздача Bootstrap и статики, API и фоновый воркер.
"""

import logging
import os
import threading
import time
import uuid
from flask import Flask, request, jsonify, send_file, send_from_directory, render_template

# Лог аналитики и воркера (файл analytics.log в корне проекта)
_log = logging.getLogger("video_gen_analytics")
_log.setLevel(logging.DEBUG)
if not _log.handlers:
    _log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analytics.log")
    _h = logging.FileHandler(_log_path, encoding="utf-8")
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _log.addHandler(_h)

from models import MODEL_RUNNERS, MODELS_META
from pipeline import run_pipeline

# Пути относительно корня проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BOOTSTRAP_DIR = os.path.join(BASE_DIR, "bootstrap-5.0.2-dist")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"), static_folder=os.path.join(BASE_DIR, "static"))

# GPU/VRAM при старте (лениво)
_gpu_name = None
_vram_gb = None


def _get_gpu_info():
    global _gpu_name, _vram_gb
    if _vram_gb is not None:
        return _gpu_name, _vram_gb
    try:
        import torch
        if torch.cuda.is_available():
            _gpu_name = torch.cuda.get_device_name(0)
            _vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        else:
            _gpu_name = None
            _vram_gb = 0.0
    except Exception:
        _gpu_name = None
        _vram_gb = 0.0
    return _gpu_name, _vram_gb


# Состояние текущей задачи
_task_lock = threading.Lock()
_current_task_id = None
_task_state = {
    "status": "idle",
    "progress": 0,
    "log": [],
    "output_path": None,
    "meta": {},
}


def _append_log(msg: str) -> None:
    with _task_lock:
        _task_state["log"].append(msg)


def _set_status(s: str) -> None:
    with _task_lock:
        _task_state["status"] = s


def _set_progress(p: int) -> None:
    with _task_lock:
        _task_state["progress"] = min(100, max(0, p))


def _set_output_path(p: str) -> None:
    with _task_lock:
        _task_state["output_path"] = p


def _set_meta(num_segments: int, prompt_preview: str, **kwargs) -> None:
    with _task_lock:
        _task_state["meta"] = {
            "num_segments": num_segments,
            "prompt_preview": (prompt_preview or "")[:500],
            **kwargs,
        }


def _reset_task(task_id: str) -> None:
    with _task_lock:
        global _current_task_id
        _current_task_id = task_id
        _task_state["status"] = "idle"
        _task_state["progress"] = 0
        _task_state["log"] = []
        _task_state["output_path"] = None
        _task_state["meta"] = {}


def _worker(prompt_ru: str, model_id: str, task_id: str) -> None:
    _log.info("worker started task_id=%s model_id=%s", task_id, model_id)
    t0 = time.time()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(0)
    except Exception:
        pass
    run_pipeline(
        prompt_ru=prompt_ru,
        model_id=model_id,
        output_dir=OUTPUTS_DIR,
        task_id=task_id,
        set_status=_set_status,
        set_progress=_set_progress,
        append_log=_append_log,
        set_output_path=_set_output_path,
        set_meta=_set_meta,
    )
    duration_sec = time.time() - t0
    with _task_lock:
        status = _task_state["status"]
        meta = _task_state.get("meta", {})
    _log.info("pipeline finished task_id=%s status=%s duration_sec=%.1f", task_id, status, time.time() - t0)
    vram_peak_gb = 0.0
    vram_peak_pct = 0.0
    try:
        import torch
        if torch.cuda.is_available():
            peak_bytes = torch.cuda.max_memory_allocated(0)
            vram_peak_gb = peak_bytes / (1024 ** 3)
            total_gb = _get_gpu_info()[1] or 0.0
            if total_gb > 0:
                vram_peak_pct = round(vram_peak_gb / total_gb * 100, 1)
    except Exception:
        pass
    if status != "done":
        _log.warning("analytics skipped: status is '%s' (not 'done') task_id=%s", status, task_id)
    else:
        try:
            import analytics
            gpu_name, vram_gb = _get_gpu_info()
            _log.debug("recording generation task_id=%s model_id=%s gpu=%s vram_gb=%.1f", task_id, model_id, gpu_name, vram_gb or 0)
            analytics.record_generation(
                task_id=task_id,
                model_id=model_id,
                gpu_name=gpu_name or "",
                vram_gb=vram_gb or 0.0,
                vram_peak_gb=vram_peak_gb,
                vram_peak_pct=vram_peak_pct,
                duration_sec=duration_sec,
                num_segments=meta.get("num_segments", 0),
                prompt_preview=meta.get("prompt_preview", "")[:500],
                avg_s_per_it=meta.get("avg_s_per_it"),
                num_frames=meta.get("num_frames"),
                video_width=meta.get("video_width"),
                video_height=meta.get("video_height"),
            )
            _log.info("analytics recorded task_id=%s model_id=%s", task_id, model_id)
        except Exception as e:
            _log.exception("analytics record_generation failed task_id=%s: %s", task_id, e)


@app.route("/bootstrap/<path:filename>")
def serve_bootstrap(filename):
    return send_from_directory(BOOTSTRAP_DIR, filename)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/models")
def api_models():
    gpu_name, vram_gb = _get_gpu_info()
    out = []
    for m in MODELS_META:
        rec = dict(m)
        min_gb = m.get("min_vram_gb")
        rec["insufficient_vram"] = bool(min_gb is not None and vram_gb < min_gb)
        out.append(rec)
    return jsonify(out)


@app.route("/api/generate", methods=["POST"])
def api_generate():
    data = request.get_json() or {}
    prompt = (data.get("prompt") or "").strip()
    model_id = (data.get("model_id") or "").strip()
    if not prompt:
        return jsonify({"error": "Промпт не указан"}), 400
    if not model_id or model_id not in MODEL_RUNNERS:
        model_id = MODELS_META[0]["id"] if MODELS_META else "stub"
    if model_id not in MODEL_RUNNERS:
        model_id = "stub"
    with _task_lock:
        if _current_task_id is not None and _task_state["status"] in ("translating", "generating", "stitching"):
            return jsonify({"error": "Генерация уже выполняется"}), 429
    task_id = str(uuid.uuid4())
    _reset_task(task_id)
    thread = threading.Thread(target=_worker, args=(prompt, model_id, task_id))
    thread.daemon = True
    thread.start()
    return jsonify({"task_id": task_id})


@app.route("/api/status")
def api_status():
    task_id = request.args.get("task_id")
    with _task_lock:
        if task_id != _current_task_id:
            return jsonify({"status": "idle", "progress": 0})
        st = _task_state.copy()
    out = {"status": st["status"], "progress": st["progress"]}
    if st["status"] == "done" and st.get("output_path") and os.path.isfile(st["output_path"]):
        out["video_url"] = "/api/video/" + task_id
    return jsonify(out)


@app.route("/api/log")
def api_log():
    task_id = request.args.get("task_id")
    with _task_lock:
        if task_id != _current_task_id:
            return jsonify({"lines": []})
        return jsonify({"lines": list(_task_state["log"])})


def _ensure_analytics_db():
    """Создаёт таблицу аналитики при любом способе запуска (flask run, gunicorn, python app.py)."""
    try:
        import analytics
        analytics.init_db()
        _log.info("analytics DB initialized path=%s", getattr(analytics, "DB_PATH", "?"))
    except Exception as e:
        _log.exception("analytics init_db failed: %s", e)


# Инициализация БД аналитики при загрузке приложения
_ensure_analytics_db()


@app.route("/api/video/<task_id>")
def api_video(task_id):
    with _task_lock:
        if task_id != _current_task_id:
            return "Not found", 404
        path = _task_state.get("output_path")
    if not path or not os.path.isfile(path):
        return "File not found", 404
    return send_file(path, as_attachment=True, download_name="video.mp4", mimetype="video/mp4")


@app.route("/analytics")
def page_analytics():
    return render_template("analytics.html")


@app.route("/api/analytics/stats")
def api_analytics_stats():
    try:
        import analytics
        data = analytics.get_stats_for_charts()
        _log.debug("api_analytics/stats rows_count=%s", len(data.get("rows", [])))
        return jsonify(data)
    except Exception as e:
        _log.exception("api_analytics/stats failed: %s", e)
        return jsonify({"by_model": [], "by_gpu": [], "rows": []})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
