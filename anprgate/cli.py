"""Командная строка ANPR-Gate."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .config import load_config
from .db import Database
from .models.color import COLOR_NAMES, REFERENCE_LAB, normalize_color_name
from .plates import canonical, is_valid_ru


def _plate_arg(text: str) -> str:
    plate = canonical(text)
    if not is_valid_ru(plate):
        print(f"Внимание: «{plate}» не соответствует формату легкового номера РФ", file=sys.stderr)
    return plate


def cmd_ui(args, cfg) -> int:
    from .ui.main_window import run_ui

    if args.source:
        cfg.source.uri = args.source
    return run_ui(cfg)


def cmd_headless(args, cfg) -> int:
    from .app import build_pipeline, load_models
    from .video import VideoSource, mask_credentials

    if args.source:
        cfg.source.uri = args.source
    db = Database(cfg.db_path)
    source = VideoSource(cfg.source_uri(), loop=False, realtime=args.realtime)
    pipeline = build_pipeline(cfg, db, load_models(cfg), source)
    events = []
    last_ts = 0.0
    try:
        for frame in source.frames():
            result = pipeline.process(frame)
            events.extend(result.events)
            last_ts = frame.ts
            if args.max_frames and frame.index + 1 >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(mask_credentials(str(exc)), file=sys.stderr)
        return 1
    events.extend(pipeline.flush(last_ts))
    for ev in events:
        pos = f"{ev.media_pos:7.2f}с" if ev.media_pos is not None else ev.ts.strftime("%H:%M:%S")
        print(f"[{pos}] #{ev.event_id} {ev.plate:<10} цвет={ev.color.name or '—':<12} "
              f"→ {ev.decision.decision.upper():<6} | {'; '.join(ev.decision.reasons)}")
        for n, stage in enumerate(ev.decision.stages, 1):
            print(f"           этап {n}. {stage.line()}")
    print(f"Итого событий: {len(events)}")
    return 0


def cmd_db(args, cfg) -> int:
    db = Database(cfg.db_path)
    if args.db_cmd == "init":
        print(f"База готова: {cfg.db_path}")
    elif args.db_cmd == "add":
        plate = _plate_arg(args.plate)
        color = normalize_color_name(args.color)
        lab = REFERENCE_LAB[color] if args.reference_lab else None
        existing = db.find_by_plate(plate)
        if existing:
            db.update_vehicle(existing.id, color=color, owner=args.owner, note=args.note, active=True,
                              color_lab=lab or existing.color_lab)
            print(f"Обновлено: {plate}")
        else:
            db.add_vehicle(plate, color, owner=args.owner, note=args.note, color_lab=lab)
            print(f"Добавлено: {plate}")
    elif args.db_cmd == "remove":
        vehicle = db.find_by_plate(canonical(args.plate))
        if not vehicle:
            print("Номер не найден")
            return 1
        db.remove_vehicle(vehicle.id)
        print(f"Удалено: {vehicle.plate}")
    elif args.db_cmd == "list":
        vehicles = db.list_vehicles()
        for v in vehicles:
            state = "активен" if v.active else "отключён"
            print(f"{v.plate:<10} {v.color:<12} {state:<9} {v.owner or ''} {v.note or ''}".rstrip())
        print(f"Всего: {len(vehicles)}")
    elif args.db_cmd == "import":
        with open(args.csv, newline="", encoding="utf-8-sig") as fh:
            rows = [
                (_plate_arg(r["plate"]), normalize_color_name(r["color"]), r.get("owner") or None, r.get("note") or None)
                for r in csv.DictReader(fh)
            ]
        print(f"Импортировано: {db.import_vehicles(rows)}")
    elif args.db_cmd == "colors":
        print(", ".join(COLOR_NAMES))
    return 0


def cmd_events(args, cfg) -> int:
    db = Database(cfg.db_path)
    for row in reversed(db.recent_events(args.limit)):
        reasons = "; ".join(json.loads(row["reasons"]))
        op = f" [оператор: {row['operator_action']}]" if row["operator_action"] else ""
        print(f"{row['ts']} #{row['id']} {row['plate'] or '—':<10} {row['color_detected'] or '—':<12} "
              f"{row['decision']:<6} {reasons}{op}")
    return 0


def cmd_eval(args, cfg) -> int:
    from .evaluate import evaluate, load_labels, summarize, write_errors
    from .models import build_plate_reader

    labels = Path(args.labels)
    samples = load_labels(labels, Path(args.images_root) if args.images_root else labels.parent)
    if args.limit:
        samples = samples[: args.limit]
    cfg.models.plate.min_plate_width_px = 0
    print(f"Изображений: {len(samples)}")
    for backend in args.backend:
        cfg.models.plate.backend = backend
        outcomes = evaluate(build_plate_reader(cfg), samples)
        print(f"\n== {backend}")
        print(f"{'группа':<10} {'n':>5} {'найден':>8} {'точно':>8} {'±1 симв':>8} {'мс':>7}")
        for row in summarize(outcomes):
            print(f"{row['group']:<10} {row['n']:>5} {row['detected']:>8.1%} {row['exact']:>8.1%} "
                  f"{row['within1']:>8.1%} {row['ms']:>7.1f}")
        errors = labels.parent / f"errors_{backend}.csv"
        write_errors(outcomes, errors)
        print(f"Ошибки: {errors}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="anprgate", description="Контроль проезда по номеру и цвету ТС")
    parser.add_argument("--config", type=Path, default=None, help="путь к anpr.yaml")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd")

    p = sub.add_parser("ui", help="окно оператора (по умолчанию)")
    p.add_argument("--source", help="mp4-файл или RTSP-адрес")
    p.set_defaults(func=cmd_ui)

    p = sub.add_parser("headless", help="обработка без интерфейса, вывод событий в консоль")
    p.add_argument("--source", help="mp4-файл или RTSP-адрес")
    p.add_argument("--realtime", action="store_true", help="воспроизводить файл в реальном темпе")
    p.add_argument("--max-frames", type=int, default=0)
    p.set_defaults(func=cmd_headless)

    p = sub.add_parser("db", help="справочник ТС")
    dbs = p.add_subparsers(dest="db_cmd", required=True)
    dbs.add_parser("init")
    a = dbs.add_parser("add")
    a.add_argument("plate")
    a.add_argument("--color", required=True)
    a.add_argument("--owner")
    a.add_argument("--note")
    a.add_argument("--reference-lab", action="store_true", help="записать типовой эталон Lab для цвета")
    r = dbs.add_parser("remove")
    r.add_argument("plate")
    dbs.add_parser("list")
    i = dbs.add_parser("import", help="CSV с колонками plate,color,owner,note")
    i.add_argument("csv")
    dbs.add_parser("colors", help="список допустимых цветов")
    p.set_defaults(func=cmd_db)

    p = sub.add_parser("eval", help="точность распознавания номеров на размеченных фото")
    p.add_argument("--labels", required=True, help="CSV (;) с колонками image, plate_num[, plate_type]")
    p.add_argument("--images-root", help="каталог, относительно которого заданы пути image")
    p.add_argument("--backend", nargs="+", default=["fast_alpr", "ru_anpr"])
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("events", help="последние события проезда")
    p.add_argument("--limit", type=int, default=30)
    p.set_defaults(func=cmd_events)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    # Под pythonw консоли нет: журнал уходит в файл, под python — как раньше в stderr.
    if sys.stderr is None:
        log_path = cfg.base_dir / "data" / "anprgate.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
            filename=str(log_path),
        )
    else:
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
    if not getattr(args, "func", None):
        args.source, args.func = None, cmd_ui
    return int(args.func(args, cfg) or 0)
