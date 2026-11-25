"""Определение цвета кузова: k-means в пространстве CIE Lab и отображение в названия цветов."""

from __future__ import annotations

import math
from typing import Optional, Tuple

import cv2
import numpy as np

from ..config import ColorModelConfig
from .base import BBox, ColorEstimate

COLOR_NAMES = (
    "белый", "серебристый", "серый", "чёрный", "красный", "оранжевый",
    "жёлтый", "зелёный", "синий", "фиолетовый", "коричневый", "бежевый",
)

# Соседние оттенки, которые камера легко путает при разном освещении.
_COMPATIBLE = {
    frozenset(p)
    for p in [
        ("белый", "серебристый"), ("серебристый", "серый"), ("серый", "чёрный"),
        ("красный", "оранжевый"), ("оранжевый", "коричневый"), ("коричневый", "бежевый"),
        ("бежевый", "белый"), ("бежевый", "серебристый"), ("синий", "фиолетовый"),
    ]
}

# Эталонные значения Lab для заполнения справочника, если эталон не снят с камеры.
REFERENCE_LAB = {
    "белый": (90.0, 0.0, 2.0), "серебристый": (68.0, 0.0, -1.0), "серый": (45.0, 0.0, -1.0),
    "чёрный": (15.0, 0.0, 0.0), "красный": (42.0, 58.0, 38.0), "оранжевый": (62.0, 38.0, 62.0),
    "жёлтый": (82.0, 2.0, 72.0), "зелёный": (45.0, -38.0, 22.0), "синий": (32.0, 12.0, -48.0),
    "фиолетовый": (35.0, 38.0, -32.0), "коричневый": (33.0, 14.0, 22.0), "бежевый": (74.0, 4.0, 18.0),
}


def normalize_color_name(name: str) -> str:
    value = (name or "").strip().lower().replace("ё", "е")
    for known in COLOR_NAMES:
        if known.replace("ё", "е") == value:
            return known
    raise ValueError(f"Неизвестный цвет «{name}». Допустимо: {', '.join(COLOR_NAMES)}")


def colors_compatible(a: str, b: str) -> bool:
    return a == b or frozenset((a, b)) in _COMPATIBLE


def delta_e(lab1: Tuple[float, float, float], lab2: Tuple[float, float, float]) -> float:
    return math.dist(lab1, lab2)


def lab_to_name(lab: Tuple[float, float, float]) -> str:
    L, a, b = lab
    chroma = math.hypot(a, b)
    if chroma < 12:
        if L > 78:
            return "белый"
        if L > 58:
            return "серебристый"
        if L > 28:
            return "серый"
        return "чёрный"
    hue = math.degrees(math.atan2(b, a)) % 360
    if 30 <= hue < 100 and L < 50 and chroma < 45:
        return "коричневый"
    if 40 <= hue < 105 and chroma < 28 and L >= 55:
        return "бежевый"
    if hue < 45 or hue >= 345:
        return "красный"
    if hue < 70:
        return "оранжевый"
    if hue < 105:
        return "жёлтый"
    if hue < 200:
        return "зелёный"
    if hue < 300:
        return "синий"
    return "фиолетовый"


def scene_color_reason(frame: np.ndarray, cfg: ColorModelConfig) -> Optional[str]:
    """Причина, по которой цвет в кадре недостоверен (ночь, ИК), иначе None."""
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA).astype(np.int16)
    if float(small.mean()) < cfg.night_brightness:
        return "недостаточная освещённость"
    channel_diff = np.abs(small[..., 0] - small[..., 1]) + np.abs(small[..., 1] - small[..., 2])
    if float(channel_diff.mean()) < cfg.ir_channel_diff:
        return "монохромный кадр (ИК-режим)"
    return None


class KMeansLabColorEstimator:
    def __init__(self, cfg: ColorModelConfig) -> None:
        self.cfg = cfg

    def estimate(self, vehicle_crop: np.ndarray, exclude: Optional[BBox] = None) -> ColorEstimate:
        h, w = vehicle_crop.shape[:2]
        if h < 24 or w < 24:
            return ColorEstimate(None, None, "слишком маленький кроп ТС")

        # Нижняя и средняя часть кузова: меньше стёкол и фона, чем в полном кропе.
        y1, y2 = int(h * 0.35), int(h * 0.92)
        x1, x2 = int(w * 0.10), int(w * 0.90)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 1
        if exclude is not None:
            ex1, ey1, ex2, ey2 = exclude
            pad_x, pad_y = int((ex2 - ex1) * 0.3), int((ey2 - ey1) * 0.6)
            mask[max(0, ey1 - pad_y):ey2 + pad_y, max(0, ex1 - pad_x):ex2 + pad_x] = 0

        lab = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2LAB)
        pixels = lab[mask.astype(bool)]
        if len(pixels) < 200:
            return ColorEstimate(None, None, "мало пикселей кузова")
        if len(pixels) > 6000:
            idx = np.random.default_rng(0).choice(len(pixels), 6000, replace=False)
            pixels = pixels[idx]

        data = pixels.astype(np.float32)
        k = max(1, min(self.cfg.clusters, len(data) // 50))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(data, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
        counts = np.bincount(labels.ravel(), minlength=k)
        shares = counts / counts.sum()
        # OpenCV хранит 8-битный Lab как L*255/100, a+128, b+128.
        labs = [(float(c[0]) * 100.0 / 255.0, float(c[1]) - 128.0, float(c[2]) - 128.0) for c in centers]

        order = list(np.argsort(-shares))
        chosen = order[0]
        # Стёкла, шины и тени — тёмные и бесцветные; это цвет кузова, только если их явное большинство.
        if self._is_dark_neutral(labs[chosen]) and shares[chosen] < self.cfg.dark_majority:
            for idx in order[1:]:
                if shares[idx] >= 0.15 and not self._is_dark_neutral(labs[idx]):
                    chosen = idx
                    break
        lab_value = labs[chosen]
        return ColorEstimate(lab_to_name(lab_value), lab_value)

    @staticmethod
    def _is_dark_neutral(lab: Tuple[float, float, float]) -> bool:
        return lab[0] < 30 and math.hypot(lab[1], lab[2]) < 15
