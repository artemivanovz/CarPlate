"""Детектор и OCR номеров РФ (ГОСТ Р 50577-2018) из shershunov/russian-license-plate-anpr.

Код инференса — third_party/ru_plate_anpr (копия из crinzon/rover-anpr), веса — weights/ru_plate.
У исходных репозиториев не указана лицензия: использование вне оценки требует разрешения автора.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np

from ..config import PlateModelConfig
from .base import PlateReading
from .vehicle_detector import resolve_torch_device

THIRD_PARTY = Path(__file__).resolve().parents[2] / "third_party" / "ru_plate_anpr"


def _ensure_rms_norm() -> None:
    """Совместимость со старыми окружениями без F.rms_norm (torch < 2.4)."""
    import torch
    import torch.nn.functional as F

    if hasattr(F, "rms_norm"):
        return

    def rms_norm(x, normalized_shape, weight=None, eps=None):
        dims = tuple(range(-len(normalized_shape), 0))
        eps = torch.finfo(x.dtype).eps if eps is None else eps
        y = x * torch.rsqrt(x.pow(2).mean(dim=dims, keepdim=True) + eps)
        return y * weight if weight is not None else y

    F.rms_norm = rms_norm


class RuPlateReader:
    def __init__(self, cfg: PlateModelConfig, weights_dir: Path) -> None:
        _ensure_rms_norm()
        if str(THIRD_PARTY) not in sys.path:
            sys.path.insert(0, str(THIRD_PARTY))
        from full_stand.config import StandConfig
        from full_stand.pipeline import PlatePipeline

        self.cfg = cfg
        device = resolve_torch_device(cfg.device)
        self.pipeline = PlatePipeline(
            StandConfig(
                detector=weights_dir / "plate_detector.pt",
                recognizer=weights_dir / "plate_recognizer.pt",
                device=device,
                confidence=cfg.detector_conf,
                use_graphs=False,
            )
        )
        self.pipeline.warmup()
        self.providers = [f"torch:{self.pipeline.device}"]

    def read(self, image: np.ndarray) -> List[PlateReading]:
        boxes, scores = self.pipeline._detect(image)
        plates = self.pipeline._read(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), boxes, scores)
        readings: List[PlateReading] = []
        for plate in plates:
            x1, y1, x2, y2 = (int(round(v)) for v in plate["box"])
            if x2 - x1 < self.cfg.min_plate_width_px or not plate["text"]:
                continue
            readings.append(
                PlateReading(
                    bbox=(x1, y1, x2, y2),
                    det_conf=float(plate["detection"]),
                    text=plate["text"],
                    ocr_conf=float(plate["confidence"]),
                    subtype=plate["subtype"],
                )
            )
        return readings
