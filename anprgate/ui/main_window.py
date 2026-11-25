"""Главное окно оператора."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QBrush, QColor
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
    QMainWindow, QMessageBox, QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ..access import DECISION_TITLES
from ..config import AppConfig
from ..db import Database
from ..pipeline import AccessEvent
from ..video import mask_credentials
from .cameras import friendly_name, remember_source
from .common import DECISION_COLORS
from .event_popup import OPERATOR_ACTIONS, EventPopup
from .source_dialog import SourceDialog
from .vehicles_dialog import VehiclesDialog
from .video_view import VideoView
from .worker import VideoWorker


class MainWindow(QMainWindow):
    COLUMNS = ("Время", "Номер", "Цвет", "Решение", "Оператор")

    def __init__(self, cfg: AppConfig, db: Database) -> None:
        super().__init__()
        self.cfg = cfg
        self.db = db
        self.events: Dict[int, AccessEvent] = {}
        self.event_rows: Dict[int, int] = {}
        self.popups: List[EventPopup] = []
        self._closing = False
        self._db_closed = False
        self.setWindowTitle("ANPR-Gate — пост оператора")
        self.resize(1500, 860)

        self.video = VideoView()
        self.video.set_placeholder("Загрузка нейросетей…")

        self.open_file_btn = QPushButton("Файл…")
        self.open_file_btn.setToolTip("Открыть видеофайл (mp4/avi/mkv/mov) — для проверки без камеры")
        self.open_rtsp_btn = QPushButton("Камера (RTSP)…")
        self.open_rtsp_btn.setToolTip(
            "Выбрать камеру: поиск по ранее подключённым RTSP-адресам\nили ввод нового адреса"
        )
        self.pause_btn = QPushButton("Пауза")
        self.pause_btn.setToolTip("Остановить и возобновить обработку текущего источника")
        self.pause_btn.setCheckable(True)
        self.stop_btn = QPushButton("Стоп")
        self.stop_btn.setToolTip("Отключиться от источника")
        self.vehicles_btn = QPushButton("Справочник ТС…")
        for b in (self.open_file_btn, self.open_rtsp_btn, self.pause_btn, self.stop_btn):
            b.setEnabled(False)
        self.open_file_btn.clicked.connect(self._open_file)
        self.open_rtsp_btn.clicked.connect(self._open_rtsp)
        self.pause_btn.toggled.connect(self._toggle_pause)
        self.stop_btn.clicked.connect(self._stop)
        self.vehicles_btn.clicked.connect(lambda: VehiclesDialog(self.db, self).exec())

        self.table = QTableWidget(0, len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels(self.COLUMNS)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.doubleClicked.connect(self._reopen_event)

        controls = QHBoxLayout()
        for b in (self.open_file_btn, self.open_rtsp_btn, self.pause_btn, self.stop_btn):
            controls.addWidget(b)
        controls.addStretch()
        controls.addWidget(self.vehicles_btn)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("Журнал проездов (двойной щелчок — карточка события)"))
        right_layout.addWidget(self.table)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.video)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 1)

        central = QWidget()
        layout = QVBoxLayout(central)
        layout.addLayout(controls)
        layout.addWidget(splitter)
        self.setCentralWidget(central)

        self.source_label = QLabel("Источник: —")
        self.perf_label = QLabel("")
        self.device_label = QLabel("")
        self.statusBar().addWidget(self.source_label, 1)
        self.statusBar().addPermanentWidget(self.perf_label)
        self.statusBar().addPermanentWidget(self.device_label)

        self.worker = VideoWorker(cfg, db)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.event_ready.connect(self._on_event)
        self.worker.status.connect(self.source_label.setText)
        self.worker.models_ready.connect(self._on_models_ready)
        self.worker.source_finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.reference_saved.connect(self._on_reference_saved)
        self.worker.reference_failed.connect(self._on_reference_failed)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.start()

    # --- Источник -----------------------------------------------------------

    def _on_models_ready(self, devices: str) -> None:
        if self._closing:
            return
        self.device_label.setText(devices)
        for b in (self.open_file_btn, self.open_rtsp_btn, self.pause_btn, self.stop_btn):
            b.setEnabled(True)
        self.video.set_placeholder("Выберите камеру — кнопка «Камера (RTSP)…» — или откройте видеофайл")
        uri = self.cfg.source_uri()
        if "://" in uri or Path(uri).exists():
            self._start(uri)
        else:
            self.source_label.setText(f"Файл из конфигурации не найден: {uri}")

    def _start(self, uri: str) -> None:
        self.pause_btn.setChecked(False)
        self.source_label.setText(f"Источник: {mask_credentials(uri)}")
        remember_source(self.cfg, uri)
        self.setWindowTitle(f"ANPR-Gate — {friendly_name(uri)}")
        self.worker.open_source(uri)

    def _open_file(self) -> None:
        start_dir = str(self.cfg.resolve("data/videos"))
        path, _ = QFileDialog.getOpenFileName(self, "Видеофайл", start_dir, "Видео (*.mp4 *.avi *.mkv *.mov);;Все файлы (*)")
        if path:
            self._start(path)

    def _open_rtsp(self) -> None:
        dialog = SourceDialog(self.cfg, self)
        if dialog.exec() and dialog.chosen_uri:
            self._start(dialog.chosen_uri)

    def _toggle_pause(self, paused: bool) -> None:
        self.pause_btn.setText("Продолжить" if paused else "Пауза")
        self.worker.set_paused(paused)

    def _stop(self) -> None:
        self.worker.stop_source()

    def _on_finished(self, name: str) -> None:
        self.source_label.setText(f"Обработка завершена: {name}")
        self.perf_label.setText("")

    def _on_failed(self, message: str) -> None:
        if self._closing:
            return
        self.source_label.setText("Ошибка")
        QMessageBox.critical(self, "ANPR-Gate", message)

    # --- Кадры и события ----------------------------------------------------

    def _on_frame(self) -> None:
        item = self.worker.latest()
        if item is None:
            return
        image, result, fps = item
        self.video.set_frame(image, result.overlays)
        if result.processed:
            self.perf_label.setText(f"{fps:.0f} кадр/с · инференс {result.infer_ms:.0f} мс")

    def _on_event(self, event: AccessEvent) -> None:
        if self._closing:
            return
        self.events[event.event_id] = event
        self.table.insertRow(0)
        for eid in self.event_rows:
            self.event_rows[eid] += 1
        self.event_rows[event.event_id] = 0
        when = f"{event.media_pos:.1f} с" if event.media_pos is not None else f"{event.ts:%H:%M:%S}"
        values = (when, event.plate or "не распознан", event.color.name or "—", DECISION_TITLES[event.decision.decision], "")
        brush = QBrush(DECISION_COLORS[event.decision.decision].lighter(170))
        for col, value in enumerate(values):
            cell = QTableWidgetItem(value)
            cell.setData(Qt.ItemDataRole.UserRole, event.event_id)
            cell.setBackground(brush)
            cell.setForeground(QBrush(QColor(0, 0, 0)))
            self.table.setItem(0, col, cell)
        self._trim_table()
        self._show_popup(event)

    def _trim_table(self) -> None:
        limit = self.cfg.ui.max_events_in_list
        while self.table.rowCount() > limit:
            row = self.table.rowCount() - 1
            eid = self.table.item(row, 0).data(Qt.ItemDataRole.UserRole)
            self.table.removeRow(row)
            self.events.pop(eid, None)
            self.event_rows.pop(eid, None)

    def _show_popup(self, event: AccessEvent) -> None:
        popup = EventPopup(event, self.db, self.cfg.ui.popup_autoclose_sec, self)
        popup.action_done.connect(self._on_operator_action)
        popup.body_reference_requested.connect(self.worker.save_body_reference)
        popup.destroyed.connect(lambda *_: self._forget_popup(popup))
        self.popups.append(popup)
        popup.adjustSize()
        self._place_popups()
        popup.show()

    def _forget_popup(self, popup: EventPopup) -> None:
        if popup in self.popups:
            self.popups.remove(popup)

    def _place_popups(self) -> None:
        screen = self.screen().availableGeometry()
        x_right, y_bottom = screen.right() - 16, screen.bottom() - 16
        for popup in reversed(self.popups[-3:]):
            size = popup.sizeHint()
            popup.move(x_right - size.width(), max(screen.top(), y_bottom - size.height()))
            y_bottom -= 40

    def _reopen_event(self, index) -> None:
        eid = self.table.item(index.row(), 0).data(Qt.ItemDataRole.UserRole)
        if eid in self.events:
            self._show_popup(self.events[eid])

    def _on_operator_action(self, event_id: int, action: str) -> None:
        row = self.event_rows.get(event_id)
        if row is not None:
            self.table.item(row, 4).setText(OPERATOR_ACTIONS[action])

    def _on_reference_saved(self, event_id: int) -> None:
        if not self._closing:
            for popup in self.popups:
                if popup.event.event_id == event_id and hasattr(popup, "save_body_btn"):
                    popup.body_reference_saved()

    def _on_reference_failed(self, event_id: int, message: str) -> None:
        if not self._closing:
            for popup in self.popups:
                if popup.event.event_id == event_id and hasattr(popup, "save_body_btn"):
                    popup.body_reference_failed(message)
                    break

    def _on_worker_finished(self) -> None:
        if self._closing:
            self.close()

    def closeEvent(self, event) -> None:
        self._closing = True
        if self.worker.isRunning():
            event.ignore()
            self.centralWidget().setEnabled(False)
            for popup in self.popups:
                popup.setEnabled(False)
            self.source_label.setText("Завершение обработки…")
            self.worker.shutdown()
            return
        if not self._db_closed:
            self.db.close()
            self._db_closed = True
        super().closeEvent(event)


def run_ui(cfg: AppConfig) -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("ANPR-Gate")
    window = MainWindow(cfg, Database(cfg.db_path))
    window.show()
    return app.exec()
