"""Детекция номерной пластины и OCR на базе FastALPR (ONNX, лицензия MIT)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from statistics import mean
from typing import List, Optional, Sequence

import numpy as np

from ..config import PlateModelConfig
from .base import PlateReading

log = logging.getLogger(__name__)


def _expose_cuda_dlls() -> None:
    """На Windows даём ONNX Runtime доступ к DLL CUDA/cuDNN 9 из torch.
    Также поддерживаем уже установленные DLL из nvidia-cudnn-cu12."""
    import importlib.util

    dirs = []
    for package, sub in (("torch", "lib"), ("nvidia.cudnn", "bin")):
        try:
            spec = importlib.util.find_spec(package)
        except ModuleNotFoundError:
            spec = None
        locations = list(spec.submodule_search_locations or []) if spec else []
        dirs += [str(Path(loc) / sub) for loc in locations if (Path(loc) / sub).is_dir()]
    if dirs:
        os.environ["PATH"] = os.pathsep.join(dirs + [os.environ.get("PATH", "")])


def onnx_providers(device: str) -> Sequence[str]:
    if device == "cpu":
        return ["CPUExecutionProvider"]
    import torch
    import onnxruntime as ort

    if torch.cuda.is_available() and "CUDAExecutionProvider" in ort.get_available_providers():
        _expose_cuda_dlls()
        # ORT 1.22 loads CUDA/cuDNN from torch on Windows and nvidia packages on Linux.
        ort.preload_dlls()
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cuda":
        log.warning("CUDAExecutionProvider недоступен, OCR и детектор номеров работают на CPU")
    return ["CPUExecutionProvider"]


def session_providers(root, depth: int = 4) -> List[str]:
    """Фактические провайдеры созданных сессий ONNX Runtime (CUDA мог не подняться)."""
    import onnxruntime as ort

    found: List[str] = []
    seen = set()

    def walk(obj, level):
        if id(obj) in seen or level > depth:
            return
        seen.add(id(obj))
        if isinstance(obj, ort.InferenceSession):
            found.append(obj.get_providers()[0])
            return
        for value in getattr(obj, "__dict__", {}).values():
            walk(value, level + 1)

    walk(root, 0)
    return found


class FastAlprPlateReader:
    def __init__(self, cfg: PlateModelConfig) -> None:
        from fast_alpr import ALPR

        self.cfg = cfg
        providers = list(onnx_providers(cfg.device))
        self.alpr = ALPR(
            detector_model=cfg.detector_model,
            detector_conf_thresh=cfg.detector_conf,
            detector_providers=providers,
            ocr_model=cfg.ocr_model,
            ocr_providers=providers,
        )
        actual = session_providers(self.alpr)
        self.providers = sorted(set(actual)) or providers
        if "CUDAExecutionProvider" in providers and "CUDAExecutionProvider" not in actual:
            log.warning("CUDA для ONNX Runtime не поднялась, номера распознаются на CPU")
        log.info("ONNX Runtime: %s", ", ".join(self.providers))

    def read(self, image: np.ndarray) -> List[PlateReading]:
        readings: List[PlateReading] = []
        for result in self.alpr.predict(image):
            box = result.detection.bounding_box
            x1, y1 = max(int(box.x1), 0), max(int(box.y1), 0)
            x2, y2 = min(int(box.x2), image.shape[1]), min(int(box.y2), image.shape[0])
            if x2 - x1 < self.cfg.min_plate_width_px or y2 <= y1:
                continue
            text, conf = self._ocr(result.ocr)
            readings.append(
                PlateReading(
                    bbox=(x1, y1, x2, y2),
                    det_conf=float(result.detection.confidence),
                    text=text,
                    ocr_conf=conf,
                )
            )
        return readings

    @staticmethod
    def _ocr(ocr) -> tuple[str, float]:
        if ocr is None:
            return "", 0.0
        conf: Optional[float]
        if isinstance(ocr.confidence, (list, tuple)):
            conf = mean(float(c) for c in ocr.confidence) if ocr.confidence else 0.0
        else:
            conf = float(ocr.confidence)
        return (ocr.text or "").replace("_", "").strip(), conf
