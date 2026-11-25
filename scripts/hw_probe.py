import os, time
import cv2
print('opencv', cv2.__version__)
for label, params in [('sw', []), ('hw', [cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY])]:
    cap = cv2.VideoCapture('data/videos/barrier_entry_clean.mp4', cv2.CAP_FFMPEG, params=params) if params else cv2.VideoCapture('data/videos/barrier_entry_clean.mp4', cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(label, 'open failed'); continue
    mode = int(cap.get(cv2.CAP_PROP_HW_ACCELERATION)) if params else -1
    n = 0; t0 = time.perf_counter()
    while n < 240:
        ok, img = cap.read()
        if not ok: break
        n += 1
    dt = time.perf_counter() - t0
    print(label, 'mode=', mode, 'fps=', round(n / dt, 1), 'shape=', img.shape if n else '-')
    cap.release()
