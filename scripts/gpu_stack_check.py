"""Что доступно GPU-стеку проекта (вызывается вручную)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    try:
        import cv2

        print("OpenCV CUDA-девайсы:", cv2.cuda.getCudaEnabledDeviceCount())
    except Exception as exc:
        print("OpenCV CUDA:", exc)
print("ORT providers:", ort.get_available_providers())