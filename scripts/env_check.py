"""Быстрая проверка окружения: версии модулей и доступность CUDA."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

print("python", sys.version.split()[0], sys.executable)

import torch
print("torch", torch.__version__, "cuda_available:", torch.cuda.is_available())
print("cuda_runtime", torch.version.cuda, "cudnn", torch.backends.cudnn.version())

import cv2
print("cv2", cv2.__version__)

import ultralytics
print("ultralytics", ultralytics.__version__)

import onnxruntime as ort
print("onnxruntime", ort.__version__, ort.get_available_providers())

try:
    import anprgate
    print("anprgate import OK")
except Exception as exc:  # noqa: BLE001
    print("anprgate import FAILED:", exc)
