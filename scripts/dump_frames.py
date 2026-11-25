"""Сохраняет несколько кадров каждой камеры для визуальной проверки."""

import os
import sys
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;8000000")

import cv2


def main() -> int:
    url_file = Path(sys.argv[1])
    out_root = Path(sys.argv[2])
    urls = [ln.strip() for ln in url_file.read_text(encoding="utf-8-sig").splitlines() if ln.strip()]
    for i, url in enumerate(urls, 1):
        out_dir = out_root / f"cam{i}"
        out_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            print(f"cam{i}: открыть не удалось")
            continue
        saved = 0
        index = 0
        while saved < 6:
            ok, image = cap.read()
            if not ok:
                continue
            if image.mean() < 3.0:
                index += 1
                continue
            index += 1
            if index % 40 != 0:
                continue
            scale = 1280 / max(image.shape[:2])
            small = cv2.resize(image, None, fx=scale, fy=scale) if scale < 1 else image
            path = out_dir / f"frame_{saved}.jpg"
            ok_enc, buf = cv2.imencode(".jpg", small)
            if ok_enc:
                buf.tofile(str(path))
                saved += 1
        cap.release()
        print(f"cam{i}: сохранено {saved} кадров в {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())