"""Общие утилиты интерфейса."""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np
from PyQt6.QtGui import QColor, QImage, QPixmap

from ..access import ALLOW, DENY, REVIEW

DECISION_COLORS = {
    ALLOW: QColor(46, 160, 67),
    DENY: QColor(207, 34, 46),
    REVIEW: QColor(212, 160, 23),
    "pending": QColor(160, 160, 160),
}


def bgr_to_qimage(image: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def bgr_to_pixmap(image: Optional[np.ndarray]) -> Optional[QPixmap]:
    if image is None or image.size == 0:
        return None
    return QPixmap.fromImage(bgr_to_qimage(image))


def lab_to_qcolor(lab: Optional[Sequence[float]]) -> Optional[QColor]:
    if lab is None:
        return None
    pixel = np.array([[[lab[0] * 255.0 / 100.0, lab[1] + 128.0, lab[2] + 128.0]]], dtype=np.float32)
    bgr = cv2.cvtColor(np.clip(pixel, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)[0, 0]
    return QColor(int(bgr[2]), int(bgr[1]), int(bgr[0]))
