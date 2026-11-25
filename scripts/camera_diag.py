"""Диагностический прогон камеры: трекинг ТС + чтение номеров, статистика по кадрам.

Аргументы: файл с URL (один на строку), число кадров, интервал полно-обработки.
Создаётся, чтобы быстро проверить связку «поток → детекция → номер» без окна оператора.
"""

import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;8000000")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anprgate.app import build_pipeline, load_models  # noqa: E402
from anprgate.config import load_config  # noqa: E402
from anprgate.db import Database  # noqa: E402
from anprgate.video import mask_credentials  # noqa: E402


def run_camera(url: str, frames: int) -> None:
    print(f"=== {mask_credentials(url)}", flush=True)
    cfg = load_config(None)
    cfg.source.uri = url
    db = Database(cfg.db_path)
    models = load_models(cfg)

    from anprgate.video import VideoSource

    source = VideoSource(url, loop=False, realtime=False)
    pipeline = build_pipeline(cfg, db, models, source)

    read_total = 0
    processed_total = 0
    detections = Counter()
    plates = Counter()
    infer_ms = []
    last_report = time.monotonic()
    for frame in source.frames():
        read_total += 1
        result = pipeline.process(frame)
        processed_total += 1 if result.processed else 0
        infer_ms.append(result.infer_ms)
        for ov in result.overlays:
            detections[ov.label] += 1
            if ov.plate:
                plates[ov.plate] += 1
        if frame.index == 0 or (time.monotonic() - last_report > 5.0):
            last_report = time.monotonic()
            print(f"   кадр {frame.index}: ТС={sum(detections.values())} "
                  f"номера={dict(plates)} инферсунку={infer_ms[-1]:.0f}мс", flush=True)
        if read_total >= frames:
            break
    print(f"   прочитано={read_total} обработано={processed_total} "
          f"детекций={dict(detections)} похожие_номера={dict(plates.most_common(5))}", flush=True)
    print(f"   средний_инференс={sum(infer_ms) / max(len(infer_ms), 1):.0f}мс", flush=True)
    source.stop()


def main() -> int:
    logging.disable(logging.INFO)
    url_file = Path(sys.argv[1])
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    urls = [ln.strip() for ln in url_file.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    for url in urls:
        try:
            run_camera(url, frames)
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())