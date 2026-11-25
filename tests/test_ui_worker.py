"""Exercise the real Qt worker without camera connections or model downloads."""

import os
import sqlite3
import threading
import time
from datetime import datetime
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from anprgate.access import decide
from anprgate.config import AppConfig
from anprgate.db import Database
from anprgate.models.base import ColorEstimate
from anprgate.models.body import pack_gallery, unpack_gallery
from anprgate.pipeline import AccessEvent
from anprgate.ui import main_window, worker
from anprgate.ui.event_popup import EventPopup


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def wait_until(qapp, condition, timeout=3.0):
    end = time.monotonic() + timeout
    while not condition() and time.monotonic() < end:
        qapp.processEvents()
        time.sleep(0.01)
    assert condition(), "Timed out waiting for the Qt worker"


@pytest.fixture
def context(tmp_path):
    cfg = AppConfig()
    cfg.base_dir = tmp_path
    cfg.models.body.backend = "fastreid"
    cfg.models.body.plate_pad = 0.67
    cfg.access.color_check = False
    db = Database(tmp_path / "test.sqlite3")
    vid = db.add_vehicle("А123ВС77", "белый")
    decision = decide("А123ВС77", True, ColorEstimate(None, None), db.list_vehicles(), cfg.access)
    crop = np.random.default_rng(1).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    event = AccessEvent(1, datetime.now(), 0.0, "test", 1, "А123ВС77", "А123ВС77",
                        0.9, True, 3, ColorEstimate(None, None, "test"), decision,
                        None, None, crop, None, vehicle_plate_bbox=(30, 70, 100, 90))
    yield cfg, db, event, vid
    try:
        db.close()
    except sqlite3.ProgrammingError:
        pass


def models(body=None):
    return SimpleNamespace(detector=SimpleNamespace(device="cpu"),
                           plate_reader=SimpleNamespace(providers=["test"]), body_embedder=body)


def test_save_uses_configured_model_and_plate_mask_in_worker_thread(qapp, context, monkeypatch):
    cfg, db, event, vid = context
    calls = []

    class Embedder:
        device = "cpu"

        def embed(self, crops, boxes):
            calls.append((threading.get_ident(), boxes, crops[0].copy()))
            return np.ones((1, 2048), np.float32)

    configured = []
    monkeypatch.setattr(worker, "load_models", lambda cfg: models())
    monkeypatch.setattr(worker, "build_body_embedder", lambda config: configured.append(config) or Embedder())
    job = worker.VideoWorker(cfg, db)
    saved = []
    errors = []
    job.reference_saved.connect(saved.append)
    job.reference_failed.connect(lambda eid, message: errors.append(message))
    job.save_body_reference(event)
    job.start()
    try:
        wait_until(qapp, lambda: bool(saved or errors))
        assert not errors
        assert saved == [event.event_id]
        assert configured == [cfg]
        assert calls[0][0] != threading.get_ident()
        assert calls[0][1] == [event.vehicle_plate_bbox]
        np.testing.assert_array_equal(calls[0][2], event.vehicle_crop)
        assert unpack_gallery(db.body_vector(vid)).shape == (1, 2048)
    finally:
        job.shutdown()
        assert job.wait(3000)


def test_failed_reference_save_keeps_previous_gallery(qapp, context, monkeypatch):
    cfg, db, event, vid = context
    old = pack_gallery(np.ones((1, 2048), np.float32))
    db.set_body_vector(vid, old)
    bad = SimpleNamespace(device="cpu", embed=lambda crops, boxes: np.full((1, 2048), np.nan))
    monkeypatch.setattr(worker, "load_models", lambda cfg: models(bad))
    job = worker.VideoWorker(cfg, db)
    errors = []
    job.reference_failed.connect(lambda eid, message: errors.append((eid, message)))
    job.save_body_reference(event)
    job.start()
    try:
        wait_until(qapp, lambda: bool(errors))
        assert errors[0][0] == event.event_id
        assert db.body_vector(vid) == old
    finally:
        job.shutdown()
        assert job.wait(3000)


def test_popup_requests_background_save_without_overwriting_database(qapp, context):
    _cfg, db, event, vid = context
    popup = EventPopup(event, db)
    requests = []
    popup.body_reference_requested.connect(requests.append)
    try:
        popup.save_body_btn.click()
        assert requests == [event]
        assert not popup.save_body_btn.isEnabled()
        assert db.body_vector(vid) is None
        popup.body_reference_saved()
        assert popup.save_body_btn.isEnabled()
    finally:
        popup.close()


def test_close_during_model_loading_waits_for_worker_before_database_close(qapp, context, monkeypatch):
    cfg, db, _event, _vid = context
    entered = threading.Event()
    release = threading.Event()

    def slow_load(cfg):
        entered.set()
        assert release.wait(5)
        return models()

    monkeypatch.setattr(worker, "load_models", slow_load)
    window = main_window.MainWindow(cfg, db)
    window.show()
    try:
        wait_until(qapp, entered.is_set)
        started = time.monotonic()
        assert not window.close()
        assert time.monotonic() - started < 1.0
        assert not window._db_closed
        assert db.list_vehicles()
        assert window.worker.isRunning()
        release.set()
        wait_until(qapp, lambda: window._db_closed)
        assert not window.worker.isRunning()
        with pytest.raises(sqlite3.ProgrammingError):
            db.list_vehicles()
    finally:
        release.set()
        window.worker.shutdown()
        assert window.worker.wait(3000)
        window.close()
