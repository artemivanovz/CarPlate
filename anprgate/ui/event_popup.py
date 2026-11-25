"""Всплывающая карточка события проезда для оператора."""

from __future__ import annotations

from typing import Optional

from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap
from PyQt6.QtWidgets import (
    QDialog, QFrame, QGridLayout, QHBoxLayout, QLabel, QMessageBox, QPushButton, QVBoxLayout, QWidget,
)

from ..access import DECISION_TITLES, DENY, STAGE_FAIL, STAGE_OK, STAGE_SKIP, STAGE_WARN
from ..db import Database
from ..pipeline import AccessEvent
from .common import DECISION_COLORS, bgr_to_pixmap, lab_to_qcolor
from .vehicles_dialog import VehicleEditDialog, save_vehicle

STAGE_COLORS = {STAGE_OK: "#1e8e3e", STAGE_WARN: "#c77700", STAGE_FAIL: "#c5221f", STAGE_SKIP: "#888888"}
OPERATOR_ACTIONS = {"confirm": "решение подтверждено", "allow": "пропущен вручную", "deny": "отказано вручную"}


def _image_label(pixmap: Optional[QPixmap], max_w: int, max_h: int, empty: str) -> QLabel:
    label = QLabel()
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setMinimumSize(min(max_w, 160), min(max_h, 60))
    label.setStyleSheet("background:#1d1f24;color:#888;border-radius:4px;")
    if pixmap is None:
        label.setText(empty)
    else:
        label.setPixmap(pixmap.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio,
                                      Qt.TransformationMode.SmoothTransformation))
    return label


class EventPopup(QDialog):
    action_done = pyqtSignal(int, str)   # event_id, operator_action
    body_reference_requested = pyqtSignal(object)

    def __init__(self, event: AccessEvent, db: Database, autoclose_sec: float = 0, parent=None) -> None:
        super().__init__(parent, Qt.WindowType.Tool | Qt.WindowType.WindowStaysOnTopHint)
        self.event = event
        self.db = db
        self.setWindowTitle(f"Событие #{event.event_id} — {event.plate or 'номер не распознан'}")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setMinimumWidth(660)

        decision = event.decision
        color = DECISION_COLORS[decision.decision]
        banner = QLabel(DECISION_TITLES[decision.decision])
        banner.setAlignment(Qt.AlignmentFlag.AlignCenter)
        banner.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        banner.setStyleSheet(f"background:{color.name()};color:white;padding:10px;border-radius:6px;")

        plate = QLabel(event.plate or "не распознан")
        plate.setAlignment(Qt.AlignmentFlag.AlignCenter)
        plate.setFont(QFont("Consolas", 26, QFont.Weight.Bold))
        plate.setStyleSheet("background:white;color:black;border:3px solid black;border-radius:6px;padding:4px 12px;")

        images = QHBoxLayout()
        images.addWidget(_image_label(bgr_to_pixmap(event.vehicle_crop), 300, 200, "нет кадра ТС"), 3)
        right = QVBoxLayout()
        right.addWidget(_image_label(bgr_to_pixmap(event.plate_crop), 200, 70, "нет кропа номера"))
        right.addWidget(plate)
        images.addLayout(right, 2)

        grid = QGridLayout()
        grid.setColumnStretch(1, 1)
        vehicle = decision.vehicle
        when = f"{event.ts:%d.%m.%Y %H:%M:%S}"
        if event.media_pos is not None:
            when += f"  (позиция в видео {event.media_pos:.1f} с)"
        rows = [
            ("Время", when),
            ("Прочитано OCR", f"{event.plate_raw}  · уверенность {event.plate_conf:.0%} · прочтений {event.votes}"),
            ("Формат РФ", "соответствует" if event.plate_valid else "НЕ соответствует"),
            ("Цвет на камере", event.color.name or f"не определён ({event.color.reason})"),
        ]
        if decision.body_score is not None:
            rows.append(("Кузов", f"близость к эталону {decision.body_score:.2f}"))
        if vehicle:
            match = "точное" if not decision.match_distance else f"нечёткое, отличий: {decision.match_distance}"
            rows += [
                ("В базе", f"{vehicle.plate} ({match})"),
                ("Цвет в базе", vehicle.color + (f" · ΔE {decision.color_delta_e:.0f}" if decision.color_delta_e is not None else "")),
                ("Владелец", vehicle.owner or "—"),
                ("Пропуск", "действует" if vehicle.active else "отключён"),
            ]
        else:
            rows.append(("В базе", "не найден"))
        for i, (name, value) in enumerate(rows):
            key = QLabel(name + ":")
            key.setStyleSheet("color:#666;")
            val = QLabel(value)
            val.setWordWrap(True)
            lines = 2 if name in ("Время", "Прочитано OCR") else 1
            val.setMinimumHeight(val.fontMetrics().lineSpacing() * lines)
            grid.addWidget(key, i, 0, Qt.AlignmentFlag.AlignTop)
            grid.addWidget(val, i, 1)
            if name == "Цвет на камере":
                swatch = self._swatch(lab_to_qcolor(event.color.lab))
                if swatch:
                    grid.addWidget(swatch, i, 2)

        stages = QVBoxLayout()
        for n, stage in enumerate(decision.stages, 1):
            label = QLabel(f"Этап {n}. {stage.line()}")
            label.setWordWrap(True)
            label.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            label.setStyleSheet(f"color:{STAGE_COLORS[stage.status]};")
            stages.addWidget(label)

        reasons = QLabel("• " + "\n• ".join(decision.reasons))
        reasons.setWordWrap(True)
        reasons.setStyleSheet(f"border-left:4px solid {color.name()};padding-left:8px;")

        buttons = QHBoxLayout()
        reference_buttons = QHBoxLayout()
        confirm = QPushButton("Подтвердить решение")
        allow = QPushButton("Пропустить вручную")
        deny = QPushButton("Отказать")
        confirm.clicked.connect(lambda: self._act("confirm"))
        allow.clicked.connect(lambda: self._act("allow"))
        deny.clicked.connect(lambda: self._act("deny"))
        for b in (confirm, allow, deny):
            buttons.addWidget(b)
        if vehicle is None:
            add = QPushButton("Добавить в базу…")
            add.clicked.connect(self._add_to_db)
            reference_buttons.addWidget(add)
        elif decision.decision == DENY and event.color.name and vehicle.color != event.color.name:
            update = QPushButton("Обновить цвет в базе…")
            update.clicked.connect(self._update_color)
            reference_buttons.addWidget(update)
        if vehicle is not None and event.vehicle_crop is not None:
            self.save_body_btn = QPushButton("Сохранить эталон кузова")
            self.save_body_btn.clicked.connect(self._save_body_reference)
            reference_buttons.addWidget(self.save_body_btn)
        confirm.setDefault(True)

        layout = QVBoxLayout(self)
        layout.addWidget(banner)
        layout.addLayout(images)
        layout.addLayout(stages)
        layout.addLayout(grid)
        layout.addWidget(reasons)
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        layout.addWidget(line)
        layout.addLayout(buttons)
        if reference_buttons.count():
            layout.addLayout(reference_buttons)

        if autoclose_sec and autoclose_sec > 0:
            QTimer.singleShot(int(autoclose_sec * 1000), self.close)

    @staticmethod
    def _swatch(qcolor) -> Optional[QWidget]:
        if qcolor is None:
            return None
        swatch = QLabel()
        swatch.setFixedSize(36, 18)
        swatch.setStyleSheet(f"background:{qcolor.name()};border:1px solid #444;")
        swatch.setToolTip(f"Lab-образец кузова: {qcolor.name()}")
        return swatch

    def _act(self, action: str) -> None:
        self.db.set_operator_action(self.event.event_id, action)
        self.action_done.emit(self.event.event_id, action)
        self.close()

    def _add_to_db(self) -> None:
        dialog = VehicleEditDialog(self, plate=self.event.plate, color=self.event.color.name,
                                   color_lab=self.event.color.lab)
        if dialog.exec() and save_vehicle(self, self.db, dialog):
            self.setWindowTitle(self.windowTitle() + " · добавлен в базу")

    def _update_color(self) -> None:
        vehicle = self.event.decision.vehicle
        dialog = VehicleEditDialog(self, vehicle)
        dialog.color.setCurrentText(self.event.color.name)
        dialog.color_lab = self.event.color.lab
        dialog.use_lab.setEnabled(self.event.color.lab is not None)
        dialog.use_lab.setChecked(self.event.color.lab is not None)
        if dialog.exec() and save_vehicle(self, self.db, dialog, vehicle.id):
            self.setWindowTitle(self.windowTitle() + " · цвет обновлён")

    def _save_body_reference(self) -> None:
        self.save_body_btn.setEnabled(False)
        self.save_body_btn.setText("Сохранение эталона…")
        self.body_reference_requested.emit(self.event)

    def body_reference_saved(self) -> None:
        self.save_body_btn.setEnabled(True)
        self.save_body_btn.setText("Сохранить эталон кузова")
        self.setWindowTitle(self.windowTitle() + " · эталон кузова сохранён")

    def body_reference_failed(self, message: str) -> None:
        self.save_body_btn.setEnabled(True)
        self.save_body_btn.setText("Сохранить эталон кузова")
        QMessageBox.warning(self, "Эталон кузова", message)
