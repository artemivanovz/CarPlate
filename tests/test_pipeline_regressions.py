"""Decisions and display coordinates must remain safe on every finalization path."""

from types import SimpleNamespace

import numpy as np
import pytest

from anprgate.access import BodyMatch
from anprgate.config import AppConfig
from anprgate.db import Database
from anprgate.models.base import ColorEstimate, PlateReading, VehicleDetection
from anprgate.models.body import pack_gallery, unpack_gallery
from anprgate.pipeline import Pipeline
from anprgate.tracking import PlateVote, TrackState
from anprgate.video import Frame


@pytest.fixture
def context(tmp_path):
    cfg = AppConfig()
    cfg.base_dir = tmp_path
    cfg.processing.frame_interval = 1
    cfg.access.color_check = False
    cfg.access.body_check = True
    db = Database(tmp_path / "test.sqlite3")
    db.add_vehicle("А123ВС77", "белый")
    yield cfg, db
    db.close()


class Detector:
    visible = True

    def track(self, image):
        return [VehicleDetection(1, (20, 20, 180, 180), "car", 0.9)] if self.visible else []


class Reader:
    def __init__(self, plates):
        self.plates = iter(plates)

    def read(self, crop):
        return [PlateReading((20, 70, 100, 95), 0.9, next(self.plates), 0.9)]


@pytest.mark.parametrize("finish", ["max_votes", "lost", "eof"])
def test_disputed_ocr_requires_review_on_every_finalization_path(context, finish):
    cfg, db = context
    variants = ["А123ВС77", "В234СЕ78", "К345МН79", "О456РТ80"] * 3
    detector = Detector()
    pipe = Pipeline(cfg, db, detector, Reader(variants), None, "test")
    pipe._body_match = lambda *args: BodyMatch(0.99, frames=4, references=1)
    image = np.zeros((200, 200, 3), np.uint8)
    events = []
    count = 12 if finish == "max_votes" else 9
    for index in range(count):
        events.extend(pipe.process(Frame(index, index / 10, image)).events)
    if finish == "lost":
        detector.visible = False
        events.extend(pipe.process(Frame(count, 4.0, image)).events)
    elif finish == "eof":
        events.extend(pipe.flush(1.0))
    assert len(events) == 1
    assert events[0].decision.decision == "review"
    assert any("согласие" in r for r in events[0].decision.reasons)
    assert db.recent_events()[0]["decision"] == "review"


@pytest.mark.parametrize("plate, expected", [("А123ВС77", "allow"), ("Х999ХХ99", "deny")])
def test_unanimous_ocr_still_allows_known_and_denies_unknown(context, plate, expected):
    cfg, db = context
    pipe = Pipeline(cfg, db, Detector(), Reader([plate] * 3), None, "test")
    pipe._body_match = lambda *args: BodyMatch(0.99)
    image = np.zeros((200, 200, 3), np.uint8)
    events = []
    for index in range(3):
        events.extend(pipe.process(Frame(index, index / 10, image)).events)
    assert len(events) == 1
    assert events[0].decision.decision == expected
    assert events[0].vehicle_plate_bbox is not None


def test_low_consensus_never_relaxes_an_existing_denial(context):
    cfg, db = context
    pipe = Pipeline(cfg, db, None, None, None, "test")
    pipe._body_match = lambda *args: BodyMatch(0.1)
    state = TrackState(1, 0, 0, (0, 0, 100, 100), "car")
    variants = ["А123ВС77", "В234СЕ78", "К345МН79", "О456РТ80"] * 3
    state.votes = [PlateVote(p, p, 0.9, True) for p in variants]
    assert pipe._finalize(state, 1.0).decision.decision == "deny"


def test_4k_vehicle_and_plate_overlays_use_original_frame_coordinates(context):
    cfg, db = context
    cfg.processing.frame_interval = 2

    class LargeDetector:
        def track(self, image):
            assert image.shape[:2] == (1080, 1920)
            return [VehicleDetection(1, (100, 100, 500, 500), "car", 0.9)]

    reader = Reader(["А123ВС77"])
    pipe = Pipeline(cfg, db, LargeDetector(), reader, None, "test")
    image = np.zeros((2160, 3840, 3), np.uint8)
    result = pipe.process(Frame(0, 0, image))
    overlay = result.overlays[0]
    assert overlay.bbox == (200, 200, 1000, 1000)
    # The vehicle crop starts at (80, 80), and the plate at (20, 70) inside it.
    assert overlay.plate_bbox == (200, 300, 360, 350)
    assert pipe.process(Frame(1, 0.1, image)).overlays == result.overlays


@pytest.mark.parametrize("blob", [
    b"bad", b"BDY1", b"BDY1" + np.array([1, 8], dtype="<i4").tobytes(),
    np.full(8, np.nan, np.float32).tobytes(),
    np.full(8, np.inf, np.float32).tobytes(),
    np.zeros(8, np.float32).tobytes(),
])
def test_bad_body_reference_becomes_review_without_stopping_pipeline(context, blob):
    cfg, db = context
    vehicle = db.list_vehicles()[0]
    db.set_body_vector(vehicle.id, blob)
    embedder = SimpleNamespace(embed=lambda crops, boxes: np.ones((1, 8), np.float32))
    pipe = Pipeline(cfg, db, None, None, None, "test", body_embedder=embedder)
    state = TrackState(1, 0, 0, (0, 0, 100, 100), "car")
    state.votes = [PlateVote(vehicle.plate, vehicle.plate, 0.9, True)] * 3
    state.body_samples = [(1.0, np.ones((32, 32, 3), np.uint8), (5, 20, 20, 28))]
    event = pipe._finalize(state, 1.0)
    assert event.decision.decision == "review"
    assert "сверка кузова не выполнена" in event.decision.reasons
    assert db.recent_events()[0]["decision"] == "review"


def test_truncated_bdy_header_is_not_interpreted_as_legacy_vector():
    with pytest.raises(ValueError):
        unpack_gallery(b"BDY1" + np.ones(3, dtype="<f4").tobytes())


@pytest.mark.parametrize("vectors", [np.zeros((1, 8)), np.full((1, 8), np.nan)])
def test_invalid_reference_cannot_be_saved(vectors):
    with pytest.raises(ValueError):
        pack_gallery(vectors)
