"""Контракты нейросетевых модулей. Конкретные реализации взаимозаменяемы через конфиг."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

import numpy as np

BBox = Tuple[int, int, int, int]  # x1, y1, x2, y2 в координатах кадра


@dataclass
class VehicleDetection:
    track_id: Optional[int]
    bbox: BBox
    label: str
    conf: float


@dataclass
class PlateReading:
    bbox: BBox
    det_conf: float
    text: str
    ocr_conf: float
    subtype: Optional[str] = None               # тип знака по ГОСТ, если модель его определяет


@dataclass
class ColorEstimate:
    name: Optional[str]                         # None — цвет определить нельзя
    lab: Optional[Tuple[float, float, float]]   # CIE Lab: L 0..100, a/b −128..127
    reason: Optional[str] = None                # причина, если name is None


class VehicleDetector(Protocol):
    def track(self, frame: np.ndarray) -> List[VehicleDetection]:
        """Детекция и трекинг ТС; track_id стабилен между кадрами."""


class PlateReader(Protocol):
    def read(self, image: np.ndarray) -> List[PlateReading]:
        """Детекция номеров и OCR на изображении (обычно — кроп ТС)."""


class ColorEstimator(Protocol):
    def estimate(self, vehicle_crop: np.ndarray, exclude: Optional[BBox] = None) -> ColorEstimate:
        """Доминирующий цвет кузова; exclude — область номера в координатах кропа."""
