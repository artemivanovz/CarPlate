"""Справочник транспортных средств: просмотр, добавление, изменение, удаление."""

from __future__ import annotations

import sqlite3
from typing import Optional, Sequence

from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
    QHeaderView, QLineEdit, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from ..db import Database, Vehicle
from ..models.color import COLOR_NAMES
from ..plates import canonical, is_valid_ru


class VehicleEditDialog(QDialog):
    def __init__(self, parent=None, vehicle: Optional[Vehicle] = None, plate: str = "",
                 color: Optional[str] = None, color_lab: Optional[Sequence[float]] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Транспортное средство")
        self.color_lab = vehicle.color_lab if vehicle else color_lab

        self.plate = QLineEdit(vehicle.plate if vehicle else plate)
        self.plate.setPlaceholderText("А123ВС77")
        self.color = QComboBox()
        self.color.addItems(COLOR_NAMES)
        current = vehicle.color if vehicle else color
        if current in COLOR_NAMES:
            self.color.setCurrentText(current)
        self.owner = QLineEdit(vehicle.owner or "" if vehicle else "")
        self.note = QLineEdit(vehicle.note or "" if vehicle else "")
        self.active = QCheckBox("Пропуск действует")
        self.active.setChecked(vehicle.active if vehicle else True)
        self.use_lab = QCheckBox("Сохранить эталон цвета, снятый с камеры")
        self.use_lab.setChecked(self.color_lab is not None)
        self.use_lab.setEnabled(self.color_lab is not None)

        form = QFormLayout()
        form.addRow("Номер", self.plate)
        form.addRow("Цвет", self.color)
        form.addRow("Владелец", self.owner)
        form.addRow("Примечание", self.note)
        form.addRow("", self.active)
        form.addRow("", self.use_lab)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addLayout(form)
        layout.addWidget(buttons)

    def _accept(self) -> None:
        plate = canonical(self.plate.text())
        if not plate:
            QMessageBox.warning(self, "Номер", "Укажите номер")
            return
        if not is_valid_ru(plate):
            answer = QMessageBox.question(
                self, "Формат номера", f"«{plate}» не соответствует формату легкового номера РФ. Сохранить?"
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.plate.setText(plate)
        self.accept()

    def values(self) -> dict:
        return {
            "plate": self.plate.text(),
            "color": self.color.currentText(),
            "owner": self.owner.text().strip() or None,
            "note": self.note.text().strip() or None,
            "active": self.active.isChecked(),
            "color_lab": self.color_lab if self.use_lab.isChecked() else None,
        }


def save_vehicle(parent, db: Database, dialog: VehicleEditDialog, vehicle_id: Optional[int] = None) -> bool:
    values = dialog.values()
    try:
        if vehicle_id is None:
            db.add_vehicle(**values)
        else:
            db.update_vehicle(vehicle_id, **values)
    except sqlite3.IntegrityError:
        QMessageBox.warning(parent, "Справочник", f"Номер {values['plate']} уже есть в базе")
        return False
    return True


class VehiclesDialog(QDialog):
    COLUMNS = ("Номер", "Цвет", "Эталон", "Владелец", "Примечание", "Статус")

    def __init__(self, db: Database, parent=None) -> None:
        super().__init__(parent)
        self.db = db
        self.setWindowTitle("Справочник транспортных средств")
        self.resize(820, 480)

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.doubleClicked.connect(self._edit)

        add, edit, remove = QPushButton("Добавить"), QPushButton("Изменить"), QPushButton("Удалить")
        add.clicked.connect(self._add)
        edit.clicked.connect(self._edit)
        remove.clicked.connect(self._remove)
        buttons = QHBoxLayout()
        for b in (add, edit, remove):
            buttons.addWidget(b)
        buttons.addStretch()

        layout = QVBoxLayout(self)
        layout.addLayout(buttons)
        layout.addWidget(self.table)
        self._vehicles = []
        self.refresh()

    def refresh(self) -> None:
        self._vehicles = self.db.list_vehicles()
        self.table.setRowCount(len(self._vehicles))
        for row, v in enumerate(self._vehicles):
            values = (v.plate, v.color, "есть" if v.color_lab else "—", v.owner or "", v.note or "",
                      "действует" if v.active else "отключён")
            for col, value in enumerate(values):
                self.table.setItem(row, col, QTableWidgetItem(value))

    def _selected(self) -> Optional[Vehicle]:
        row = self.table.currentRow()
        return self._vehicles[row] if 0 <= row < len(self._vehicles) else None

    def _add(self) -> None:
        dialog = VehicleEditDialog(self)
        if dialog.exec() and save_vehicle(self, self.db, dialog):
            self.refresh()

    def _edit(self) -> None:
        vehicle = self._selected()
        if not vehicle:
            return
        dialog = VehicleEditDialog(self, vehicle)
        if dialog.exec() and save_vehicle(self, self.db, dialog, vehicle.id):
            self.refresh()

    def _remove(self) -> None:
        vehicle = self._selected()
        if not vehicle:
            return
        answer = QMessageBox.question(self, "Удаление", f"Удалить {vehicle.plate} из справочника?")
        if answer == QMessageBox.StandardButton.Yes:
            self.db.remove_vehicle(vehicle.id)
            self.refresh()
