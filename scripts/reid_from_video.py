"""Пересобрать эталон кузова (reid_vector) для ТС из эталонного ролика.

Запуск:
  .venv\\Scripts\\python.exe scripts\\reid_from_video.py <видео> <номер>
Например:
  python scripts\\reid_from_video.py data\\videos\\etalon_crossover.mp4 А001АА799

Процедура: детектор ТС нарезает кропы, номерная табличка исключается из контура,
ShapeBodyEmbedder считает векторы. В базу пишется галерея кадров одного кузова:
если в ролике проезжает вторая машина, её векторы в эталон не усредняются.
Галерея привязана к vehicle_id — кузова разных записей не смешиваются.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from anprgate.config import load_config  # noqa: E402
from anprgate.db import Database  # noqa: E402
from anprgate.models import build_plate_reader, build_vehicle_detector  # noqa: E402
from anprgate.models.body import ShapeBodyEmbedder, pack_gallery, score_gallery, select_prototypes, unpack_gallery  # noqa: E402
from anprgate.video import VideoSource  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    video, plate = sys.argv[1], sys.argv[2].replace(" ", "").upper()
    cfg = load_config()
    db = Database(cfg.db_path)
    vehicle = db.find_by_plate(plate)
    if vehicle is None:
        print(f"Номер {plate} не найден в базе")
        return 1

    old = db.body_vector(vehicle.id)
    print(f"ТС #{vehicle.id} {vehicle.plate} ({vehicle.color}) — ролик {video}")

    detector = build_vehicle_detector(cfg)
    reader = build_plate_reader(cfg)
    embedder = ShapeBodyEmbedder(cfg.models.body)

    source = VideoSource(str(Path(video)), loop=False, realtime=False)
    vectors, frames_seen = [], 0
    for frame in source.frames():
        frames_seen += 1
        if frames_seen % 5:              # детектор не на каждом кадре
            continue
        result = detector.model.predict(
            frame.image,
            classes=cfg.models.vehicle.classes,
            conf=0.3,
            imgsz=cfg.models.vehicle.imgsz,
            device=detector.device,
            verbose=False,
        )
        boxes = result[0].boxes
        if boxes is None or len(boxes) == 0:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        x1, y1, x2, y2 = (int(round(v)) for v in xyxy[int(np.argmax((xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])))])
        crop = frame.image[y1:y2, x1:x2]
        if min(crop.shape[:2]) < 40:
            continue
        readings = reader.read(crop)     # bbox таблички — чтобы сеть не видела номер
        plate_box = readings[0].bbox if readings else None
        batch = embedder.embed([crop], [plate_box])
        if len(batch):
            vectors.append(batch[0])

    if not vectors:
        print(f"Кропов не нашлось ({frames_seen} кадров)")
        return 1
    stack = np.stack(vectors)
    gallery = select_prototypes(stack)
    if len(gallery) < len(stack):
        print(f"отброшено кадров другого контура: {len(stack) - len(gallery)} из {len(stack)}")

    if old:
        old_gallery = unpack_gallery(old)
        if old_gallery.shape[1] == gallery.shape[1]:
            print(f"старый эталон: близость кадров {score_gallery(stack, old_gallery):.2f}")
        else:
            print("старый эталон: другой размер вектора (галерея пересчитана)")
    print(
        f"новая галерея: {len(gallery)} эталонов из {len(vectors)} кропов, "
        f"близость своих кадров {score_gallery(gallery, gallery):.2f}"
    )

    db.set_body_vector(vehicle.id, pack_gallery(gallery))
    print(f"Записана галерея ({gallery.shape[1]} значений × {len(gallery)}) для {vehicle.plate}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())