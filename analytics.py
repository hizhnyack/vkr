# -*- coding: utf-8 -*-
"""Система аналитики генераций: SQLite, запись отчётов, данные для графиков."""

import logging
import os
import sqlite3
from typing import Any, Dict, List

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("ANALYTICS_DB", os.path.join(BASE_DIR, "analytics.db"))

_log = logging.getLogger("video_gen_analytics")


def _get_conn():
    return sqlite3.connect(DB_PATH)


def init_db() -> None:
    try:
        with _get_conn() as c:
            c.execute("""
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                model_id TEXT NOT NULL,
                gpu_name TEXT,
                vram_gb REAL,
                vram_peak_gb REAL,
                vram_peak_pct REAL,
                duration_sec REAL,
                num_segments INTEGER,
                prompt_preview TEXT,
                avg_s_per_it REAL,
                num_frames INTEGER,
                video_width INTEGER,
                video_height INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
            for col_def in [
                "vram_peak_gb REAL", "vram_peak_pct REAL",
                "avg_s_per_it REAL", "num_frames INTEGER", "video_width INTEGER", "video_height INTEGER",
            ]:
                try:
                    c.execute("ALTER TABLE generations ADD COLUMN " + col_def)
                except sqlite3.OperationalError:
                    pass
        _log.debug("init_db ok path=%s", DB_PATH)
    except Exception as e:
        _log.exception("init_db failed path=%s: %s", DB_PATH, e)
        raise


def record_generation(
    task_id: str,
    model_id: str,
    gpu_name: str,
    vram_gb: float,
    vram_peak_gb: float = 0.0,
    vram_peak_pct: float = 0.0,
    duration_sec: float = 0.0,
    num_segments: int = 0,
    prompt_preview: str = "",
    avg_s_per_it: float = None,
    num_frames: int = None,
    video_width: int = None,
    video_height: int = None,
) -> None:
    preview = (prompt_preview or "")[:500]
    try:
        with _get_conn() as c:
            c.execute(
                """INSERT INTO generations (task_id, model_id, gpu_name, vram_gb, vram_peak_gb, vram_peak_pct, duration_sec, num_segments, prompt_preview, avg_s_per_it, num_frames, video_width, video_height)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (task_id, model_id, gpu_name, vram_gb, vram_peak_gb, vram_peak_pct, duration_sec, num_segments, preview, avg_s_per_it, num_frames, video_width, video_height),
            )
        _log.debug("record_generation ok task_id=%s path=%s", task_id, DB_PATH)
    except Exception as e:
        _log.exception("record_generation failed task_id=%s path=%s: %s", task_id, DB_PATH, e)
        raise


def get_stats_for_charts() -> Dict[str, Any]:
    with _get_conn() as c:
        c.row_factory = sqlite3.Row
        # По моделям: среднее время, количество
        def row_to_dict(r):
            return dict(zip(r.keys(), r)) if hasattr(r, "keys") else dict(r)

        cur = c.execute("""
            SELECT model_id, AVG(duration_sec) AS avg_duration, COUNT(*) AS count
            FROM generations GROUP BY model_id
        """)
        by_model = [row_to_dict(r) for r in cur.fetchall()]

        # По GPU: среднее время, количество
        cur = c.execute("""
            SELECT gpu_name, AVG(duration_sec) AS avg_duration, COUNT(*) AS count
            FROM generations WHERE gpu_name IS NOT NULL AND gpu_name != ''
            GROUP BY gpu_name
        """)
        by_gpu = [row_to_dict(r) for r in cur.fetchall()]

        # Сырые записи для таблицы (последние 200)
        cur = c.execute("""
            SELECT task_id, model_id, gpu_name, vram_gb, vram_peak_gb, vram_peak_pct, duration_sec, num_segments, prompt_preview, avg_s_per_it, num_frames, video_width, video_height, created_at
            FROM generations ORDER BY id DESC LIMIT 200
        """)
        rows = [row_to_dict(r) for r in cur.fetchall()]

    return {"by_model": by_model, "by_gpu": by_gpu, "rows": rows}
