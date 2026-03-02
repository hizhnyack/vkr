#!/usr/bin/env python3
"""
Веб-интерфейс пайплайна симуляции дорожных сценариев.
Запуск из корня проекта: python web/app.py  или  flask --app web.app run
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_file, url_for
from werkzeug.exceptions import RequestEntityTooLarge

WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent
UPLOADS_DIR = WEB_DIR / "uploads"
OUTPUTS_DIR = WEB_DIR / "outputs"
BOOTSTRAP_DIR = ROOT / "bootstrap-5.0.2-dist"
CONFIG_PATH = ROOT / "config_default.yaml"
LOG_MAX_LINES = 500
MAX_CONTENT_MB = 500
ALLOWED_EXTENSIONS = {"mp4", "mkv", "mpeg", "avi"}
LOG_FILE_PATH = ROOT / "pipeline.log"

# Лог-буфер в памяти (последние N строк)
_log_buffer: deque = deque(maxlen=LOG_MAX_LINES)
_log_lock = threading.Lock()
_log_file = None


def _init_log_file() -> None:
    global _log_file
    try:
        _log_file = open(LOG_FILE_PATH, "w", encoding="utf-8")
        _log_file.write(f"--- Сессия запущена {datetime.now().isoformat()} ---\n")
        _log_file.flush()
    except OSError as e:
        logging.warning("Не удалось открыть файл логов %s: %s", LOG_FILE_PATH, e)
        _log_file = None


_init_log_file()
# Состояние задач: job_id -> { "status": "pending"|"running"|"done"|"error", "output_path": Path|None, "error": str|None }
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _append_log(line: str) -> None:
    with _log_lock:
        _log_buffer.append(line)
        if _log_file is not None:
            try:
                _log_file.write(line + "\n")
                _log_file.flush()
            except OSError:
                pass


def _get_logs(tail: int = 200) -> list[str]:
    with _log_lock:
        return list(_log_buffer)[-tail:]


def _run_pipeline(job_id: str, video_path: Path, task_text: str, output_path: Path) -> None:
    with _jobs_lock:
        _jobs[job_id]["status"] = "running"
    _append_log(f"[{job_id}] Запуск пайплайна: video={video_path.name}, task={task_text[:50]}...")

    try:
        cmd = [
            sys.executable,
            str(ROOT / "run_pipeline.py"),
            "--video", str(video_path),
            "--task", task_text,
            "--output", str(output_path),
            "--config", str(CONFIG_PATH),
        ]
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in proc.stdout:
            line = line.rstrip()
            if "[PIPELINE_STAGE]" in line:
                stage_text = line.split("[PIPELINE_STAGE]", 1)[-1].strip()
                with _jobs_lock:
                    _jobs[job_id]["stage"] = stage_text
            _append_log(f"[{job_id}] {line}")
        proc.wait()
        if proc.returncode != 0:
            with _jobs_lock:
                _jobs[job_id]["status"] = "error"
                _jobs[job_id]["error"] = f"Пайплайн завершился с кодом {proc.returncode}"
            _append_log(f"[{job_id}] Ошибка: код выхода {proc.returncode}")
            return
        with _jobs_lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["output_path"] = output_path
        _append_log(f"[{job_id}] Готово. Результат: {output_path.name}")
    except Exception as e:
        with _jobs_lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(e)
        _append_log(f"[{job_id}] Исключение: {e}")
        logging.exception("Pipeline run failed")


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__, static_folder=str(WEB_DIR / "static"), template_folder=str(WEB_DIR / "templates"))
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-in-production")


# Раздача Bootstrap 5.0.2 из папки проекта
@app.route("/bootstrap/<path:filename>")
def serve_bootstrap(filename: str):
    path = (BOOTSTRAP_DIR / filename).resolve()
    try:
        if not path.is_relative_to(BOOTSTRAP_DIR.resolve()) or not path.exists():
            return "", 404
    except AttributeError:
        if not str(path).startswith(str(BOOTSTRAP_DIR.resolve())) or not path.exists():
            return "", 404
    return send_file(path, mimetype="application/octet-stream")


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(e):
    _append_log(f"[server] Ошибка: файл слишком большой (макс. {MAX_CONTENT_MB} МБ)")
    return jsonify({"error": f"Файл слишком большой. Максимум {MAX_CONTENT_MB} МБ."}), 413


@app.errorhandler(500)
def handle_500(e):
    _append_log(f"[server] Ошибка 500: {e}")
    return jsonify({"error": "Внутренняя ошибка сервера"}), 500


@app.route("/")
def index():
    from flask import render_template
    return render_template("index.html")


@app.post("/run")
def run():
    if "video" not in request.files:
        _append_log("[server] POST /run: нет поля video")
        return jsonify({"error": "Не выбран файл видео"}), 400
    if "task" not in request.form or not request.form["task"].strip():
        _append_log("[server] POST /run: не задан промпт")
        return jsonify({"error": "Введите текст задачи (промпт)"}), 400

    file = request.files["video"]
    task_text = request.form["task"].strip()
    if file.filename == "":
        return jsonify({"error": "Не выбран файл"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": f"Допустимые форматы: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    job_id = str(uuid.uuid4())
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    ext = file.filename.rsplit(".", 1)[1].lower()
    upload_path = UPLOADS_DIR / f"{job_id}.{ext}"
    output_path = OUTPUTS_DIR / f"{job_id}.mp4"

    try:
        file.save(str(upload_path))
    except Exception as e:
        _append_log(f"[server] Ошибка сохранения файла: {e}")
        return jsonify({"error": "Не удалось сохранить файл"}), 500

    with _jobs_lock:
        _jobs[job_id] = {"status": "pending", "output_path": None, "error": None, "stage": None}

    thread = threading.Thread(target=_run_pipeline, args=(job_id, upload_path, task_text, output_path))
    thread.daemon = True
    thread.start()
    _append_log(f"[server] Создана задача {job_id}")
    return jsonify({"job_id": job_id, "status": "pending"})


@app.get("/api/status/<job_id>")
def api_status(job_id: str):
    with _jobs_lock:
        if job_id not in _jobs:
            return jsonify({"error": "Задача не найдена"}), 404
        job = _jobs[job_id].copy()
    status = job["status"]
    out = {"job_id": job_id, "status": status}
    if job.get("stage") is not None:
        out["stage"] = job["stage"]
    if status == "done" and job.get("output_path"):
        out["download_url"] = url_for("download", job_id=job_id, _external=False)
    if status == "error" and job.get("error"):
        out["error"] = job["error"]
    return jsonify(out)


@app.get("/api/logs")
def api_logs():
    tail = request.args.get("tail", type=int, default=100)
    tail = min(max(1, tail), 500)
    job_id = request.args.get("job_id")
    lines = _get_logs(tail=tail)
    if job_id:
        lines = [ln for ln in lines if f"[{job_id}]" in ln]
    return jsonify({"lines": lines})


@app.get("/api/report/<job_id>")
def api_report(job_id: str):
    with _jobs_lock:
        if job_id not in _jobs or _jobs[job_id]["status"] != "done":
            return jsonify({"error": "Задача не найдена или не завершена"}), 404
        output_path = _jobs[job_id].get("output_path")
    if not output_path:
        return jsonify({"error": "Нет пути к результату"}), 404
    path = Path(output_path)
    report_path = path.parent / (path.name + ".report.json")
    if not report_path.exists():
        return jsonify({"error": "Отчёт не найден"}), 404
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return jsonify({"error": "Ошибка чтения отчёта"}), 500
    return jsonify(data)


@app.get("/download/<job_id>")
def download(job_id: str):
    with _jobs_lock:
        if job_id not in _jobs or _jobs[job_id]["status"] != "done":
            return jsonify({"error": "Файл недоступен"}), 404
        output_path = _jobs[job_id].get("output_path")
    if not output_path or not Path(output_path).exists():
        return jsonify({"error": "Файл не найден"}), 404
    return send_file(
        output_path,
        as_attachment=True,
        download_name=f"result_{job_id}.mp4",
        mimetype="video/mp4",
    )


# Логирование ошибок Flask в общий буфер
class BufferHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            _append_log(f"[server] {msg}")
        except Exception:
            self.handleError(record)


logging.getLogger("werkzeug").setLevel(logging.WARNING)
app_log = logging.getLogger("flask.app")
app_log.addHandler(BufferHandler())
app_log.setLevel(logging.INFO)


if __name__ == "__main__":
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    _append_log("[server] Веб-сервер запущен")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
