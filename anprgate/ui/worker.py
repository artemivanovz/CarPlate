"""Фоновый поток: чтение видео и работа конвейера вне потока интерфейса."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Optional, Tuple

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from ..app import build_pipeline, load_models
from ..config import AppConfig
from ..db import Database
from ..models import build_body_embedder
from ..models.body import make_body_reference
from ..pipeline import FrameResult
from ..video import VideoSource, mask_credentials

log = logging.getLogger(__name__)


class VideoWorker(QThread):
    frame_ready = pyqtSignal()                 # новый кадр доступен через latest()
    event_ready = pyqtSignal(object)           # AccessEvent
    status = pyqtSignal(str)
    models_ready = pyqtSignal(str)             # описание устройств инференса
    source_finished = pyqtSignal(str)
    failed = pyqtSignal(str)
    reference_saved = pyqtSignal(int)           # event_id
    reference_failed = pyqtSignal(int, str)

    def __init__(self, cfg: AppConfig, db: Database) -> None:
        super().__init__()
        self.cfg = cfg
        self.db = db
        self._lock = threading.Lock()
        self._latest: Optional[Tuple[np.ndarray, FrameResult, float]] = None
        self._pending_source: Optional[str] = None
        self._source: Optional[VideoSource] = None
        self._paused = threading.Event()
        self._running = True
        self._pending_references = deque()

    # --- Управление из потока интерфейса ----------------------------------

    def open_source(self, uri: str) -> None:
        with self._lock:
            self._pending_source = uri
        if self._source:
            self._source.stop(wait=False)
        self._paused.clear()

    def stop_source(self) -> None:
        with self._lock:
            self._pending_source = None
        if self._source:
            self._source.stop(wait=False)
        self._paused.clear()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()
            if self._source:
                self._source.reset_clock()

    def shutdown(self) -> None:
        self._running = False
        self.requestInterruption()
        with self._lock:
            self._pending_references.clear()
        self.stop_source()

    def save_body_reference(self, event) -> None:
        vehicle = event.decision.vehicle
        if not self._running or vehicle is None or event.vehicle_crop is None:
            self.reference_failed.emit(event.event_id, "нет кадра или записи ТС")
            return
        with self._lock:
            self._pending_references.append(
                (event.event_id, vehicle.id, event.vehicle_crop.copy(), event.vehicle_plate_bbox)
            )

    def _save_pending_references(self, models) -> None:
        while self._running:
            with self._lock:
                if not self._pending_references:
                    return
                event_id, vehicle_id, crop, plate_bbox = self._pending_references.popleft()
            try:
                if self.db.get_vehicle(vehicle_id) is None:
                    raise ValueError("запись ТС уже удалена")
                if models.body_embedder is None:
                    models.body_embedder = build_body_embedder(self.cfg)
                reference = make_body_reference(models.body_embedder, crop, plate_bbox)
                self.db.set_body_vector(vehicle_id, reference)
            except Exception as exc:
                message = mask_credentials(str(exc))
                log.error("Эталон кузова не сохранён: %s", message)
                self.reference_failed.emit(event_id, message)
            else:
                self.reference_saved.emit(event_id)

    def latest(self) -> Optional[Tuple[np.ndarray, FrameResult, float]]:
        with self._lock:
            item, self._latest = self._latest, None
        return item

    # --- Поток обработки --------------------------------------------------

    def run(self) -> None:
        if not self._running:
            return
        try:
            self.status.emit("Загрузка нейросетей…")
            models = load_models(self.cfg)
            if not self._running:
                return
            devices = f"ТС: {models.detector.device}; номера: {', '.join(models.plate_reader.providers)}"
            if models.body_embedder is not None:
                devices += f"; кузов: {models.body_embedder.device}"
            self.models_ready.emit(devices)
        except Exception as exc:  # noqa: BLE001 — показываем оператору любую ошибку загрузки
            log.exception("Ошибка загрузки моделей")
            self.failed.emit(f"Ошибка загрузки моделей: {exc}")
            return

        while self._running:
            self._save_pending_references(models)
            with self._lock:
                uri, self._pending_source = self._pending_source, None
            if not uri:
                time.sleep(0.05)
                continue
            self._play(uri, models)

    def _play(self, uri: str, models) -> None:
        source = None
        try:
            source = VideoSource(uri, loop=self.cfg.source.loop, realtime=self.cfg.source.realtime)
            self._source = source
            pipeline = build_pipeline(self.cfg, self.db, models, source)
            self.status.emit(f"Источник: {source.display_name}")
            fps_window, last_ts = [], 0.0
            for frame in source.frames():
                while self._paused.is_set() and self._running and not source.stopped:
                    self._save_pending_references(models)
                    time.sleep(0.05)
                if not self._running or source.stopped:
                    break
                self._save_pending_references(models)
                result = pipeline.process(frame)
                for event in result.events:
                    self.event_ready.emit(event)
                now = time.perf_counter()
                fps_window = [t for t in fps_window if now - t < 1.0] + [now]
                with self._lock:
                    self._latest = (frame.image, result, float(len(fps_window)))
                self.frame_ready.emit()
                last_ts = frame.ts
            if self._running and not source.is_stream:
                for event in pipeline.flush(last_ts):
                    self.event_ready.emit(event)
            self.source_finished.emit(source.display_name)
        except Exception as exc:  # noqa: BLE001
            message = mask_credentials(str(exc))
            log.error("Ошибка обработки источника: %s", message)
            self.failed.emit(message)
        finally:
            self._source = None
            if source is not None:
                source.stop()   # идемпотентно: гарантирует остановку потока-ридера RTSP
