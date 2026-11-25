from pathlib import Path

import numpy as np
import pytest

from dataclasses import replace

from anprgate.access import ALLOW, DENY, REVIEW, BodyMatch, decide
from anprgate.config import AccessConfig, ColorModelConfig, ProcessingConfig, load_config
from anprgate.db import Database, Vehicle
from anprgate.models.base import ColorEstimate
from anprgate.models.color import KMeansLabColorEstimator, lab_to_name, normalize_color_name, scene_color_reason
from anprgate.plates import canonical, is_valid_ru, levenshtein
from anprgate.tracking import PlateVote, TrackState


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("A123BC77", "А123ВС77"),
        ("a 123 bc 777", "А123ВС777"),
        ("0123BC77", "О123ВС77"),      # 0 на месте буквы
        ("A12OBC77", "А120ВС77"),      # О на месте цифры
        ("A123BC7T", "А123ВС77"),
        ("AA795755", "АА795755"),      # жёлтый номер такси не ломается легковым шаблоном
        ("AA79S77", "АА79577"),
    ],
)
def test_canonical(raw, expected):
    assert canonical(raw) == expected
    assert is_valid_ru(expected)


def test_invalid_format():
    assert not is_valid_ru("АБ123")
    assert not is_valid_ru(canonical("1234AB77"))


def test_levenshtein():
    assert levenshtein("А123ВС77", "А123ВС77") == 0
    assert levenshtein("А123ВС77", "А128ВС77") == 1
    assert levenshtein("А123ВС77", "А123ВС777") == 1
    assert levenshtein("А123ВС77", "Х999ХХ99", limit=1) == 2


@pytest.fixture()
def db(tmp_path: Path):
    database = Database(tmp_path / "t.sqlite3")
    yield database
    database.close()


def _color(name, lab=None):
    return ColorEstimate(name, lab or (50.0, 0.0, 0.0))


def test_decide_rules(db):
    cfg = AccessConfig()
    db.add_vehicle("А123ВС77", "белый", owner="Иванов")
    db.add_vehicle("К555КК799", "чёрный", active=False)
    vehicles = db.list_vehicles()

    assert decide("А123ВС77", True, _color("белый"), vehicles, cfg).decision == ALLOW
    assert decide("А123ВС77", True, _color("серебристый"), vehicles, cfg).decision == ALLOW  # соседний оттенок
    assert decide("А123ВС77", True, _color("красный"), vehicles, cfg).decision == DENY
    assert decide("Х001ХХ77", True, _color("белый"), vehicles, cfg).decision == DENY
    assert decide("К555КК799", True, _color("чёрный"), vehicles, cfg).decision == DENY
    assert decide("А128ВС77", True, _color("белый"), vehicles, cfg).decision == REVIEW        # нечёткое
    assert decide("А123ВС77", True, ColorEstimate(None, None, "ИК"), vehicles, cfg).decision == REVIEW
    assert decide(None, False, _color("белый"), vehicles, cfg).decision == DENY


def test_decide_two_stages(db):
    cfg = AccessConfig()
    db.add_vehicle("Н555НН98", "серый")
    vehicles = db.list_vehicles()

    def statuses(plate, color):
        return [s.status for s in decide(plate, True, _color(color), vehicles, cfg).stages]

    assert statuses("Н555НН98", "серый") == ["ok", "ok"]
    assert statuses("Н555НН98", "красный") == ["ok", "fail"]
    assert statuses("Х001ХХ77", "серый") == ["fail", "skip"]   # цвет не проверяется без номера


def test_db_roundtrip(db):
    vid = db.add_vehicle("А123ВС77", "синий", color_lab=(32, 12, -48))
    assert db.find_by_plate("А123ВС77").color_lab == (32.0, 12.0, -48.0)
    db.update_vehicle(vid, color="серый", active=False)
    v = db.get_vehicle(vid)
    assert v.color == "серый" and not v.active
    assert db.remove_vehicle(vid)
    assert db.list_vehicles() == []


def test_color_names():
    assert lab_to_name((92, 0, 1)) == "белый"
    assert lab_to_name((12, 1, 1)) == "чёрный"
    assert lab_to_name((45, 60, 40)) == "красный"
    assert lab_to_name((30, 10, -50)) == "синий"
    assert normalize_color_name("Черный") == "чёрный"
    with pytest.raises(ValueError):
        normalize_color_name("перламутровый")


def test_color_estimator_on_synthetic_car():
    crop = np.full((200, 300, 3), (40, 40, 200), dtype=np.uint8)  # BGR красный
    est = KMeansLabColorEstimator(ColorModelConfig()).estimate(crop)
    assert est.name == "красный"


def test_dark_glass_does_not_override_white_body():
    est = KMeansLabColorEstimator(ColorModelConfig())
    white_with_glass = np.full((200, 300, 3), 225, dtype=np.uint8)
    white_with_glass[70:116, :, :] = 25          # тёмное стекло ~40 % анализируемой области (строки 70..184)
    assert est.estimate(white_with_glass).name in ("белый", "серебристый")
    black_car = np.full((200, 300, 3), 20, dtype=np.uint8)
    black_car[170:185, :, :] = 200               # хромированная полоса
    assert est.estimate(black_car).name == "чёрный"


def test_scene_reason_detects_ir():
    cfg = ColorModelConfig()
    gray = np.full((360, 640, 3), 120, dtype=np.uint8)
    assert scene_color_reason(gray, cfg) is not None
    dark = np.full((360, 640, 3), (10, 5, 20), dtype=np.uint8)
    assert scene_color_reason(dark, cfg) is not None


def test_voting_prefers_consistent_valid_plate():
    state = TrackState(1, 0.0, 0.0, (0, 0, 10, 10), "car")
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    for plate, conf, valid in [("А123ВС77", 0.9, True), ("А123ВС7", 0.95, False), ("А123ВС77", 0.8, True)]:
        state.add_vote(PlateVote(plate, plate, conf, valid), img, img, img, (0, 0, 10, 10))
    vote, count, share = state.leader()
    assert vote.plate == "А123ВС77" and count == 2


def test_config_loads():
    cfg = load_config()
    assert cfg.processing.min_votes >= 1
    assert cfg.models.vehicle.weights.endswith(".pt")


@pytest.fixture()
def listed_vehicle():
    return Vehicle(1, "А123ВС77", "белый", None, "Иванов", None, True)


@pytest.fixture()
def day_access():
    """Цвет включён, кузов включён — дыра с переписыванием отказа видна."""
    return AccessConfig(color_check=True, body_check=True, fuzzy_max_distance=1, fuzzy_decision="allow")


@pytest.mark.parametrize(
    "plate, valid, color_name, active, expected",
    [
        ("А123ВС77", True, "белый", True, ALLOW),
        ("А123ВС77", True, "красный", True, DENY),
        ("Х001ХХ77", True, "белый", True, DENY),
        (None, False, "белый", True, DENY),
        ("А123ВС77", True, "белый", False, DENY),
    ],
    ids=["exact-allow", "color-deny", "unknown-deny", "unread-deny", "inactive-deny"],
)
def test_decide_matrix_without_body(plate, valid, color_name, active, expected):
    vehicle = Vehicle(1, "А123ВС77", "белый", None, None, None, active)
    decision = decide(plate, valid, _color(color_name), [vehicle], AccessConfig())
    assert decision.decision == expected


def test_fuzzy_config_allow_stays_review(listed_vehicle, day_access):
    """BUG-R-FUZZY: нечёткое совпадение не открывает проезд, даже если yaml говорит allow."""
    decision = decide("А128ВС77", True, _color("белый"), [listed_vehicle], day_access)
    assert decision.match_distance == 1
    assert decision.decision == REVIEW
    assert any("нечёткое" in reason for reason in decision.reasons)


def test_fuzzy_config_deny_stays_review(listed_vehicle):
    cfg = AccessConfig(fuzzy_decision="deny", fuzzy_max_distance=1, body_check=False)
    decision = decide("А128ВС77", True, _color("белый"), [listed_vehicle], cfg)
    assert decision.decision == REVIEW


def test_body_match_does_not_override_color_deny(listed_vehicle, day_access):
    """BUG-R-BODY: высокая близость кузова не отменяет отказ по цвету."""
    body = BodyMatch(score=0.91, frames=2, references=1)
    decision = decide("А123ВС77", True, _color("красный"), [listed_vehicle], day_access, body)
    assert decision.decision == DENY
    assert any(reason.startswith("цвет ") for reason in decision.reasons)
    assert decision.decision != ALLOW


def test_body_failure_is_journaled_and_not_allow(listed_vehicle, day_access):
    """BUG-R-EMBED: сбой сверки виден в reasons и не остаётся тихим allow."""
    body = BodyMatch(None, reason="сверка кузова не выполнена")
    decision = decide("А123ВС77", True, _color("белый"), [listed_vehicle], day_access, body)
    assert decision.decision == REVIEW
    assert "сверка кузова не выполнена" in decision.reasons
    body_stage = next(stage for stage in decision.stages if stage.title == "Кузов")
    assert body_stage.status != "skip"


def test_missing_embedder_is_not_labeled_as_missing_reference(db, listed_vehicle):
    """BUG-R-MASK: отсутствие сети не маскируется под «нет эталона»."""
    from anprgate.pipeline import Pipeline
    from anprgate.tracking import TrackState

    cfg = load_config()
    cfg.access.body_check = True
    pipe = Pipeline(cfg, db, None, None, None, source_name="test", body_embedder=None)
    state = TrackState(1, 0.0, 0.0, (0, 0, 40, 40), "car")
    match = pipe._body_match(state, listed_vehicle.plate, [listed_vehicle])
    assert match.reason is not None
    assert "эталона" not in match.reason
    assert "не выполнена" in match.reason


def test_shape_mismatch_is_a_failed_check_not_a_missing_frame(db):
    """BUG-R-SHAPE: чужая размерность вектора не называется «нет кадра»."""
    from anprgate.pipeline import Pipeline
    from anprgate.tracking import TrackState

    vid = db.add_vehicle("А123ВС77", "белый")
    vector = np.zeros(8, dtype=np.float32).tobytes()
    with db._lock:
        db._conn.execute("ALTER TABLE vehicles ADD COLUMN reid_vector BLOB")
        db._conn.execute("UPDATE vehicles SET reid_vector = ? WHERE id = ?", (vector, vid))
        db._conn.commit()

    class WrongShape:
        def embed(self, crops, plates):
            return np.zeros((1, 3), dtype=np.float32)

    cfg = load_config()
    cfg.access.body_check = True
    pipe = Pipeline(cfg, db, None, None, None, source_name="test", body_embedder=WrongShape())
    crop = np.zeros((32, 32, 3), dtype=np.uint8)
    state = TrackState(1, 0.0, 0.0, (0, 0, 40, 40), "car")
    state.body_samples = [(1.0, crop, None)]
    match = pipe._body_match(state, "А123ВС77", db.list_vehicles())
    assert match.reason is not None
    assert "кадра" not in match.reason
    assert "не выполнена" in match.reason


def _paint(color, kind):
    import cv2
    image = np.full((220, 340, 3), 90, np.uint8)
    if kind == "coupe":
        cv2.ellipse(image, (180, 130), (120, 40), 0, 0, 360, color, -1)
        cv2.ellipse(image, (155, 92), (58, 26), 0, 0, 360, color, -1)
        cv2.circle(image, (90, 162), 16, (20, 20, 20), -1)
        cv2.circle(image, (270, 162), 16, (20, 20, 20), -1)
    else:
        cv2.rectangle(image, (40, 80), (300, 175), color, -1)
        cv2.rectangle(image, (80, 40), (210, 90), color, -1)
    return image


def test_shape_vector_ignores_paint_and_keeps_silhouette():
    """Один и тот же контур в другом цвете и при другой яркости ближе, чем другая геометрия."""
    from anprgate.models.body import shape_vector

    red = shape_vector(_paint((40, 40, 210), "coupe"))
    gray = shape_vector(_paint((150, 150, 150), "coupe"))
    dark = shape_vector((_paint((40, 40, 210), "coupe").astype(np.float32) * 0.35).astype(np.uint8))
    van = shape_vector(_paint((40, 40, 210), "van"))
    same_color = float(red @ gray)
    same_night = float(red @ dark)
    other_shape = float(red @ van)
    assert same_color > 0.85
    assert same_night > 0.75
    assert other_shape < same_color - 0.15
