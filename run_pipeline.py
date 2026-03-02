#!/usr/bin/env python3
"""
Точка входа пайплайна симуляции дорожных сценариев.
Использование:
  python run_pipeline.py --video input.mp4 --task "пешеход переходит дорогу" --output out.mp4
  python run_pipeline.py --video input.mp4 --task-file task.txt --output out.mp4 [--export-annotations] [--config config.yaml]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Добавляем src в путь при запуске из корня проекта
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("pipeline")
# Меньше шума от библиотек загрузки моделей
for _name in ("diffusers", "transformers", "httpx"):
    logging.getLogger(_name).setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description="Симуляция дорожных сценариев на видео")
    parser.add_argument("--video", required=True, type=Path, help="Путь к входному видео (mkv/mp4/mpeg)")
    parser.add_argument("--task", type=str, default=None, help="Текст задачи (альтернатива --task-file)")
    parser.add_argument("--task-file", type=Path, default=None, help="Файл с текстом задачи")
    parser.add_argument("--output", required=True, type=Path, help="Путь к выходному видео")
    parser.add_argument("--config", type=Path, default=ROOT / "config_default.yaml", help="Путь к YAML-конфигу")
    parser.add_argument("--export-annotations", action="store_true", help="Экспорт разметки в JSON после генерации")
    args = parser.parse_args()

    task_text = args.task
    if task_text is None and args.task_file is not None:
        if not args.task_file.exists():
            logger.error("Файл задачи не найден: %s", args.task_file)
            sys.exit(1)
        task_text = args.task_file.read_text(encoding="utf-8").strip()
    if not task_text:
        logger.error("Задайте задачу через --task или --task-file")
        sys.exit(1)

    if not args.video.exists():
        logger.error("Видео не найдено: %s", args.video)
        sys.exit(1)

    config = Config.from_yaml(args.config)
    logger.info("Конфиг загружен: %s", args.config)

    # 1) Видео I/O и метаданные
    print("[PIPELINE_STAGE] Метаданные и кадры (ffprobe/decord)", flush=True)
    from video_io import VideoReader, VideoMetadata
    reader = VideoReader(args.video)
    metadata = reader.metadata
    logger.info("Метаданные видео: %sx%s, %.1f fps, %.1f сек", metadata.width, metadata.height, metadata.fps, metadata.duration_sec)

    # 2) Парсинг задачи
    print("[PIPELINE_STAGE] LLM: текст → JSON сценария", flush=True)
    from scenario_parser import parse_task_text, Scenario
    scenario = parse_task_text(task_text, config=config)
    logger.info("Сценарий: %d событий", len(scenario.events))

    # 3) Анализ сцены по выбранным кадрам
    print("[PIPELINE_STAGE] Сэмпл кадров", flush=True)
    from scene_analysis import SceneAnalyzer, SceneDescription
    sampled = reader.sample_frames(5)
    if not sampled:
        logger.warning("Нет кадров в видео")
        scene_descriptions = []
    else:
        print("[PIPELINE_STAGE] VLM: время суток, погода", flush=True)
        analyzer = SceneAnalyzer(
            device=config.device,
            load_vlm_4bit=getattr(config.vlm, "load_in_4bit", True),
        )
        print("[PIPELINE_STAGE] Depth: карта глубины", flush=True)
        scene_descriptions = analyzer.analyze_frames(sampled)
        logger.info("Анализ сцены: время=%s, погода=%s", scene_descriptions[0].time_of_day if scene_descriptions else "n/a", scene_descriptions[0].weather if scene_descriptions else "n/a")

    # 4) Планирование вставки
    print("[PIPELINE_STAGE] Планирование вставок", flush=True)
    from synthesis import InsertionPlanner
    planner = InsertionPlanner(metadata.width, metadata.height, metadata.fps, len(reader))
    insertion_events = planner.plan(scenario.events, scene_descriptions)
    logger.info("Запланировано вставок: %d", len(insertion_events))
    if insertion_events and scenario.events:
        ev0 = insertion_events[0]
        sev0 = scenario.events[0]
        logger.info(
            "Первое событие: %s, кадры %d–%d (%.1f–%.1f с)",
            ev0.object_type, ev0.start_frame, ev0.end_frame,
            sev0.interval.start_sec, sev0.interval.end_sec,
        )

    # 5) Читаем все кадры (или по частям для длинных видео)
    frames = []
    for i in range(len(reader)):
        frames.append(reader.get_frame(i))
    logger.info("Прочитано кадров: %d", len(frames))

    # 6) Генерация и композитинг (упрощённо: по ключевым кадрам генерируем, остальное — без изменений или интерполяция)
    print("[PIPELINE_STAGE] Генерация патчей (SD + ControlNet)", flush=True)
    from synthesis import ContentGenerator, CompositePipeline
    gen = ContentGenerator(
        device=config.device,
        num_inference_steps=config.diffusion.num_inference_steps,
        guidance_scale=config.diffusion.guidance_scale,
        max_resolution=config.diffusion.max_resolution,
    )
    time_of_day = scene_descriptions[0].time_of_day if scene_descriptions else "day"
    weather = scene_descriptions[0].weather if scene_descriptions else "clear"
    depth_for_frame = {}
    if scene_descriptions and scene_descriptions[0].get_depth_map() is not None:
        for desc in scene_descriptions:
            if desc.frame_index is not None:
                depth_for_frame[desc.frame_index] = desc.get_depth_map()

    def get_patch_for(ev_idx, frame_index, ev, frame):
        if ev_idx >= len(insertion_events):
            return None
        local_idx = frame_index - ev.start_frame
        if local_idx < 0 or (ev.bboxes and local_idx >= len(ev.bboxes)):
            return None
        bbox = ev.bboxes[local_idx]
        depth = depth_for_frame.get(frame_index)
        if depth is None and scene_descriptions:
            # Ближайший кадр с глубиной
            for d in scene_descriptions:
                if d.get_depth_map() is not None:
                    depth = d.get_depth_map()
                    break
        prompt = f"{ev.object_type} in street scene"
        try:
            return gen.generate_for_region(prompt, frame, bbox, depth, time_of_day, weather)
        except Exception as e:
            logger.info("Генерация патча не удалась: %s", e)
            return None

    comp_cfg = getattr(config, "composite", None) or object()
    use_color_matching = getattr(comp_cfg, "use_color_matching", False)
    debug_draw_border = getattr(comp_cfg, "debug_draw_border", False)
    debug_border_thickness = getattr(comp_cfg, "debug_border_thickness", 3)
    composite = CompositePipeline(
        use_color_matching=use_color_matching,
        debug_draw_border=debug_draw_border,
        debug_border_thickness=debug_border_thickness,
    )
    out_frames = composite.run(frames, insertion_events, get_patch_for)
    print("[PIPELINE_STAGE] Композитинг", flush=True)
    logger.info("Композитинг выполнен, кадров: %d", len(out_frames))

    # 7) Кодирование выхода
    print("[PIPELINE_STAGE] Запись видео (ffmpeg)", flush=True)
    from export import export_video
    args.output.parent.mkdir(parents=True, exist_ok=True)
    export_video(out_frames, args.output, metadata)
    logger.info("Видео записано: %s", args.output)

    # Отчёт: что добавлено на видео и в какие кадры
    import json
    report_path = args.output.parent / (args.output.name + ".report.json")
    report_events = []
    fps = getattr(metadata, "fps", None) or 1.0
    for i, ins_ev in enumerate(insertion_events):
        ev_data = {
            "object_type": ins_ev.object_type,
            "start_frame": ins_ev.start_frame,
            "end_frame": ins_ev.end_frame,
            "start_sec": round(ins_ev.start_frame / fps, 1),
            "end_sec": round(ins_ev.end_frame / fps, 1),
        }
        if i < len(scenario.events):
            sev = scenario.events[i]
            ev_data["event_type"] = getattr(sev.event_type, "value", str(sev.event_type))
            ev_data["description"] = getattr(sev, "description", None) or None
        report_events.append(ev_data)
    summary_parts = []
    for ev in report_events:
        summary_parts.append(f"{ev['object_type']} (кадры {ev['start_frame']}–{ev['end_frame']}, {ev['start_sec']}–{ev['end_sec']} с)")
    report = {
        "fps": fps,
        "events": report_events,
        "summary": "Добавлено {} событий: {}.".format(len(report_events), "; ".join(summary_parts)) if report_events else "Вставок нет.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Отчёт сохранён: %s", report_path)

    # 8) Опциональная разметка
    if args.export_annotations:
        print("[PIPELINE_STAGE] Разметка (YOLO)", flush=True)
        from export import export_annotations_if_requested
        json_path = args.output.with_suffix(args.output.suffix + ".annotations.json")
        export_annotations_if_requested(args.output, output_json_path=json_path)
        logger.info("Разметка сохранена: %s", json_path)

    logger.info("Готово.")


if __name__ == "__main__":
    main()
