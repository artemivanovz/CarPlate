"""Сверка кузовов на роликах: свой эталон, чужая машина, ночь и снег.

Запуск из каталога проекта:
  .venv\\Scripts\\python.exe scripts\\body_separation_check.py
  .venv\\Scripts\\python.exe scripts\\body_separation_check.py --write --gray-plate А001АА799 --red-plate Н991КК797

--write записывает галерею серого кроссовера и красного купе в их записи базы.
Чужой кузов в галерею не усредняется: select_prototypes оставляет один контур.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from anprgate.config import load_config  # noqa: E402
from anprgate.db import Database  # noqa: E402
from anprgate.plates import canonical  # noqa: E402
from anprgate.models import build_plate_reader, build_vehicle_detector  # noqa: E402
from anprgate.models.body import (  # noqa: E402
    ShapeBodyEmbedder,
    pack_gallery,
    score_gallery,
    select_prototypes,
)

# Окна роликов: серый кроссовер, красное купе, подмена кузова, ночь и другая погода.
CLIPS = (
    ("серый эталон", "gemini_b.mp4", 0.00, 0.95),
    ("серый другой номер", "gemini_swap_demo.mp4", 0.72, 0.98),
    ("седан тот же номер", "etalon_crossover.mp4", 0.15, 0.85),
    ("седан в подмене", "gemini_swap_demo.mp4", 0.02, 0.22),
    ("красное купе день", "gemini_generated_video_c5431bdf.mp4", 0.12, 0.88),
    ("красное купе снег", "gemini_generated_video_3dd2ab9d.mp4", 0.12, 0.88),
    ("красное купе снег 2", "gemini_generated_video_e5d7328c.mp4", 0.12, 0.88),
    ("купе сумерки другая краска", "gemini_generated_video_993f6f42.mp4", 0.12, 0.88),
    ("ночь чужая", "ds07p_night_clean.mp4", 0.25, 0.75),
    ("ночь КПП", "b2530_night_clean.mp4", 0.25, 0.75),
    ("день чужая", "ds07p_day_clean.mp4", 0.3, 0.6),
    ("джип подмена серый", "plate_swap_demo.mp4", 0.30, 0.58),
    ("джип подмена белый", "plate_swap_demo.mp4", 0.75, 0.96),
)


def _largest_crop(frame: np.ndarray, detector, classes, imgsz):
    result = detector.model.predict(
        frame, classes=classes, conf=0.3, imgsz=imgsz, device=detector.device, verbose=False,
    )
    boxes = result[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.cpu().numpy()
    area = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    x1, y1, x2, y2 = (int(round(v)) for v in xyxy[int(np.argmax(area))])
    height, width = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 40 or y2 - y1 < 40:
        return None
    return frame[y1:y2, x1:x2]


def _samples(path: Path, start: float, end: float, detector, reader, cfg, limit: int = 8):
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return []
    positions = np.linspace(int(total * start), max(int(total * start), int(total * end) - 1), limit)
    crops = []
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos))
        ok, frame = cap.read()
        if not ok:
            continue
        crop = _largest_crop(frame, detector, cfg.models.vehicle.classes, cfg.models.vehicle.imgsz)
        if crop is None:
            continue
        readings = reader.read(crop)
        plate_box = readings[0].bbox if readings else None
        crops.append((crop, plate_box))
    cap.release()
    return crops


def _as_night(crop: np.ndarray) -> np.ndarray:
    image = (crop.astype(np.float32) * 0.22).astype(np.uint8)
    height, width = image.shape[:2]
    radius = max(4, width // 18)
    cv2.circle(image, (int(width * 0.32), int(height * 0.58)), radius, (255, 255, 255), -1)
    cv2.circle(image, (int(width * 0.68), int(height * 0.58)), radius, (255, 255, 255), -1)
    return image


def reference_vehicles(db, gray_plate: str, red_plate: str):
    gray = db.find_by_plate(canonical(gray_plate))
    red = db.find_by_plate(canonical(red_plate))
    if gray is None or red is None or not gray.active or not red.active:
        raise ValueError("Оба номера должны существовать в базе и иметь активный пропуск")
    if gray.id == red.id:
        raise ValueError("Укажите две разные записи ТС")
    return gray, red


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--gray-plate", help="номер записи серого кроссовера")
    parser.add_argument("--red-plate", help="номер записи красного купе")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    if args.write and not (args.gray_plate and args.red_plate):
        parser.error("для --write явно укажите --gray-plate и --red-plate")
    cfg = load_config(args.config)
    if args.write:
        db = Database(cfg.db_path)
        try:
            reference_vehicles(db, args.gray_plate, args.red_plate)
        except ValueError as exc:
            print(exc)
            return 1
        finally:
            db.close()
    detector = build_vehicle_detector(cfg)
    reader = build_plate_reader(cfg)
    embedder = ShapeBodyEmbedder(cfg.models.body)
    root = cfg.resolve("data/videos")

    banks = {}
    for label, name, start, end in CLIPS:
        path = root / name
        if not path.exists():
            print(f"нет файла: {path}")
            banks[label] = np.empty((0, 0), np.float32)
            continue
        crops = _samples(path, start, end, detector, reader, cfg)
        if not crops:
            print(f"{label}: детектор не нашёл ТС")
            banks[label] = np.empty((0, 0), np.float32)
            continue
        vectors = embedder.embed([crop for crop, _box in crops], [box for _crop, box in crops])
        banks[label] = vectors
        ratio = float(np.median([crop.shape[1] / max(crop.shape[0], 1) for crop, _box in crops]))
        print(f"кадров {label}: {len(vectors)} пропорция {ratio:.2f}")

    gray_bank = banks.get("серый эталон")
    red_bank = banks.get("красное купе день")
    if gray_bank is None or len(gray_bank) == 0 or red_bank is None or len(red_bank) == 0:
        print("Нет эталонных кропов серого кроссовера или красного купе")
        return 1
    gray = select_prototypes(gray_bank)
    red = select_prototypes(red_bank)

    night_crops = _samples(root / "gemini_b.mp4", 0.05, 0.95, detector, reader, cfg, limit=6)
    if night_crops:
        darkened = [_as_night(crop) for crop, _box in night_crops]
        banks["серый эталон ночь (контроль)"] = embedder.embed(darkened, [box for _c, box in night_crops])

    print()
    print(f"{'ролик':<32} {'кадры':>5} {'свой':>7} {'серый':>7} {'красн.':>7}")
    for label, _name, _a, _b in list(CLIPS) + [("серый эталон ночь (контроль)", "", 0, 0)]:
        vectors = banks.get(label)
        if vectors is None or len(vectors) == 0:
            continue
        own = float(score_gallery(vectors, select_prototypes(vectors)))
        vs_gray = float(score_gallery(vectors, gray))
        vs_red = float(score_gallery(vectors, red))
        print(f"{label:<32} {len(vectors):5d} {own:7.2f} {vs_gray:7.2f} {vs_red:7.2f}")

    if not args.write:
        print("\nБаза не изменена. Повторите с --write, чтобы записать галереи.")
        return 0

    db = Database(cfg.db_path)
    try:
        gray_vehicle, red_vehicle = reference_vehicles(db, args.gray_plate, args.red_plate)
        db.set_body_vector(gray_vehicle.id, pack_gallery(gray))
        db.set_body_vector(red_vehicle.id, pack_gallery(red))
        print(f"Записан серый кроссовер → {gray_vehicle.plate} ({len(gray)} векторов)")
        print(f"Записано красное купе → {red_vehicle.plate} ({len(red)} векторов)")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
