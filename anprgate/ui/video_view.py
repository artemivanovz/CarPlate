"""Виджет видео с разметкой треков, номеров и решений."""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen
from PyQt6.QtWidgets import QSizePolicy, QWidget

from ..pipeline import TrackOverlay
from .common import DECISION_COLORS, bgr_to_qimage

STATUS_TEXT = {"allow": "ПРОПУСК", "deny": "ОТКАЗ", "review": "ПРОВЕРКА", "pending": ""}


class VideoView(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image: Optional[QImage] = None
        self._overlays: List[TrackOverlay] = []
        self._placeholder = "Откройте видеофайл или RTSP-поток"

    def set_placeholder(self, text: str) -> None:
        self._placeholder = text
        self.update()

    def set_frame(self, frame: np.ndarray, overlays: List[TrackOverlay]) -> None:
        self._image = bgr_to_qimage(frame)
        self._overlays = overlays
        self.update()

    def clear(self) -> None:
        self._image, self._overlays = None, []
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 24))
        if self._image is None:
            painter.setPen(QColor(170, 170, 170))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return

        iw, ih = self._image.width(), self._image.height()
        scale = min(self.width() / iw, self.height() / ih)
        dw, dh = iw * scale, ih * scale
        ox, oy = (self.width() - dw) / 2, (self.height() - dh) / 2
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawImage(QRectF(ox, oy, dw, dh), self._image)

        def rect(box) -> QRectF:
            x1, y1, x2, y2 = box
            return QRectF(ox + x1 * scale, oy + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale)

        font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        for ov in self._overlays:
            color = DECISION_COLORS.get(ov.status, DECISION_COLORS["pending"])
            painter.setPen(QPen(color, 2))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            vr = rect(ov.bbox)
            painter.drawRect(vr)
            if ov.plate_bbox:
                painter.setPen(QPen(QColor(0, 200, 255), 2))
                painter.drawRect(rect(ov.plate_bbox))

            parts = [f"#{ov.track_id}", ov.plate or "номер…"]
            if ov.color:
                parts.append(ov.color)
            if ov.body_score is not None:
                parts.append(f"кузов {ov.body_score:.2f}")
            if STATUS_TEXT.get(ov.status):
                parts.append(STATUS_TEXT[ov.status])
            text = "  ".join(parts)
            tw, th = metrics.horizontalAdvance(text) + 10, metrics.height() + 4
            label = QRectF(vr.left(), max(0.0, vr.top() - th), tw, th)
            painter.fillRect(label, color)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(label, Qt.AlignmentFlag.AlignCenter, text)
