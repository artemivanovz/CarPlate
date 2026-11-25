"""Проба RTSP-потоков: открыть, прочитать кадры, сообщить разрешение/FPS.

Аргументы: файл со списком URL (по одному на строку) и число кадров на поток.
Пути к URL из файла, чтобы не светить пароли в аргументах командной строки.
"""

import os
import sys
import time
from pathlib import Path

TRANSPORT = os.environ.get("RTSP_TRANSPORT", "tcp")
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{TRANSPORT}|stimeout;8000000"

import cv2


def probe(url: str, frames: int) -> str:
    started = time.monotonic()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return "FAIL: открыть не удалось"
    ok_count = 0
    width = height = 0
    fps = 0.0
    first_frame_ms = None
    last = time.monotonic()
    for _ in range(frames):
        ok, image = cap.read()
        now = time.monotonic()
        if not ok:
            continue
        if first_frame_ms is None:
            first_frame_ms = (now - started) * 1000
            height, width = image.shape[:2]
        ok_count += 1
        last = now
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fps = actual_fps
    wall = last - started
    cap.release()
    return (f"OK кадров={ok_count}/{frames} {width}x{height} fps_src={fps:.1f} "
            f"фпс_реальный={ok_count / wall:.1f} первый_кадр_через={first_frame_ms:.0f}мс"
            if ok_count and first_frame_ms else f"FAIL: кадров прочитано {ok_count}")


def main() -> int:
    url_file = Path(sys.argv[1])
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    print(f"probe: файл={url_file.name} кадров={frames} транспорт={TRANSPORT}", flush=True)
    for url in url_file.read_text(encoding="utf-8-sig").splitlines():
        url = url.strip()
        if not url or url.startswith("#"):
            continue
        print(url.split("//")[0] + "//" + url.split("@")[-1])
        print(probe(url, frames))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())