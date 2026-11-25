"""Диалог выбора камеры: поиск по известным RTSP-источникам или ввод нового адреса."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QVBoxLayout,
)

from ..config import AppConfig
from ..video import mask_credentials
from .cameras import (
    discover_seeds, filter_sources, forget_source, friendly_name, is_manual_uri, known_sources,
    load_recents,
)

URL_ROLE = Qt.ItemDataRole.UserRole


class SourceDialog(QDialog):
    """Список известных камер с живым поиском; новая строка поиска тоже принимается как адрес."""

    def __init__(self, cfg: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.cfg = cfg
        self.setWindowTitle("Выбор камеры / источника")
        self.setMinimumWidth(520)

        self.search = QLineEdit()
        self.search.setPlaceholderText("Поиск: адрес, имя камеры или порт — например «14.24» или «Ch101»")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._refresh)

        self.label = QLabel("Известные камеры:")
        self.list = QListWidget()
        self.list.itemDoubleClicked.connect(lambda _item: self._accept())
        self.list.itemSelectionChanged.connect(self._update_buttons)

        self.hint = QLabel(
            "Можно вставить новый адрес rtsp://… прямо в строку поиска.\n"
            "Двойной щелчок по камере — подключение."
        )
        self.hint.setStyleSheet("color: #666;")

        self.remove_btn = QPushButton("Удалить из списка")
        self.remove_btn.clicked.connect(self._remove_selected)
        self.cancel_btn = QPushButton("Отмена")
        self.cancel_btn.clicked.connect(self.reject)
        self.connect_btn = QPushButton("Подключиться")
        self.connect_btn.setDefault(True)
        self.connect_btn.clicked.connect(self._accept)

        buttons = QHBoxLayout()
        buttons.addWidget(self.remove_btn)
        buttons.addStretch()
        buttons.addWidget(self.cancel_btn)
        buttons.addWidget(self.connect_btn)

        layout = QVBoxLayout(self)
        layout.addWidget(self.search)
        layout.addWidget(self.label)
        layout.addWidget(self.list, 1)
        layout.addWidget(self.hint)
        layout.addLayout(buttons)

        self._all: List[str] = known_sources(cfg)
        self._seeds = set(discover_seeds(cfg))
        self.chosen_uri: Optional[str] = None

        self.search.setFocus()
        self._refresh()

    # --- наполнение и поиск ---------------------------------------------------

    def _refresh(self) -> None:
        query = self.search.text()
        manual = is_manual_uri(query)
        sources = self._all if manual else filter_sources(self._all, query)
        self.label.setText("Новый адрес (вставлен в поиск):" if manual else "Известные камеры:")
        self.list.clear()
        for uri in sources:
            item = QListWidgetItem(friendly_name(uri))
            item.setData(URL_ROLE, uri)
            item.setToolTip(mask_credentials(uri))
            self.list.addItem(item)
        if self.list.count():
            self.list.setCurrentRow(0)
        self._update_buttons()

    def _update_buttons(self) -> None:
        uri = self._current_uri()
        self.connect_btn.setEnabled(uri is not None)
        item = self.list.currentItem()
        selected = item.data(URL_ROLE) if item else None
        removable = selected is not None and selected in load_recents(self.cfg)
        self.remove_btn.setEnabled(removable)
        self.remove_btn.setToolTip(
            "Автоподсказка из cam*url.txt — удаляется вместе с файлом" if selected in self._seeds
            else "Убрать камеру из списка недавних подключений"
        )

    def _current_uri(self) -> Optional[str]:
        query = self.search.text().strip()
        if is_manual_uri(query):
            return query
        item = self.list.currentItem()
        return item.data(URL_ROLE) if item else None

    # --- действия -------------------------------------------------------------

    def _accept(self) -> None:
        uri = self._current_uri()
        if uri:
            self.chosen_uri = uri
            self.accept()

    def _remove_selected(self) -> None:
        item = self.list.currentItem()
        if not item:
            return
        uri = item.data(URL_ROLE)
        if uri in self._seeds:
            return  # подсказки из cam*url.txt удаляются вместе с файлом
        forget_source(self.cfg, uri)
        self._all = known_sources(self.cfg)
        self._refresh()