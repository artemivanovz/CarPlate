"""Проба: каким декодером открывается камера 101 и с какой скоростью. Вызывается вручную."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2  # noqa: E402

from anprgate import video  # noqa: E402

url_file = sys.argv[1] if len(sys.argv) > 1 else "column miss"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 60
with open(url_file, encoding="utf-8-sig") as fh:
    url = fh.readline().strip()

opened = video._open_capture(url, True)
if opened is None:
    raise SystemExit("open failed")
cap, kind = opened
mode = int(cap.get(cv2.CAP_PROP_HW_ACCELERATION) or 0)
print("kind:", kind, "backend mode:", mode, flush=True)
t0 = time.perf_counter()
ok_frames = 0
for _ in range(n):
    ok, _image = cap.read()
    ok_frames += 1 if ok else 0
dt = time.perf_counter() - t0
print(f"frames ok={ok_frames}/{n}, rate={ok_frames / dt:.1f} fps", flush=True)
cap.release()