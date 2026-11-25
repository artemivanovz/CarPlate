"""Сравнение темпа декодирования RTSP-потока: программно и с аппаратным ускорением.

Аргументы: файл с URL и число кадров. Показывает, поспевает ли клиент за камерой.
"""

import os
import sys
import time

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;8000000")

import cv2  # noqa: E402


def main() -> int:
    url = open(sys.argv[1], encoding="utf-8-sig").readline().strip()
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    variants = [("sw", []), ("hw", [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])]
    for label, params in variants:
        if params:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG, params=params)
        else:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(label, "открыть не удалось", flush=True)
            continue
        mode = int(cap.get(cv2.CAP_PROP_HW_ACCELERATION)) if params else -1
        started = None
        count = 0
        while count < frames:
            ok, image = cap.read()
            if not ok:
                continue
            if started is None:
                started = time.perf_counter()
            count += 1
        dt = time.perf_counter() - started
        print(label, "mode=", mode, "fps_decode=", round(count / dt, 1),
              "shape=", image.shape, flush=True)
        cap.release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())