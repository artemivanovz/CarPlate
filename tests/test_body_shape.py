"""Галереи кузова: свой эталон близко к 1, чужой кузов в него не входит."""

import cv2
import numpy as np

from anprgate.models.body import (
    FACE_DIM,
    SHAPE_DIM,
    SILHOUETTE_DIM,
    l2_normalize,
    crop_marked_vehicle,
    pack_gallery,
    score_gallery,
    select_prototypes,
    shape_vector,
    unpack_gallery,
)
from anprgate.pipeline import safe_plate_token


def _crop(rng: np.random.Generator, center_pattern: bool) -> np.ndarray:
    img = np.full((96, 96, 3), 30, np.uint8)
    if center_pattern:
        img[8:88, 8:88, :] = np.tile(np.linspace(30, 210, 80).astype(np.uint8), (80, 1))[:, :, None]
    fringe = max(1, int(round(0.03 * 96)))
    noise = (rng.random((96, 96, 3)) * 255).astype(np.uint8)
    img[:fringe] = noise[:fringe]
    img[-fringe:] = noise[-fringe:]
    img[:, :fringe] = noise[:, :fringe]
    img[:, -fringe:] = noise[:, -fringe:]
    return img


def _vehicle(kind: str, night: bool = False, snow: bool = False, plate_x: int = 0) -> np.ndarray:
    """Синтетический кузов: высокий джип или низкое купе, без опоры на краску."""
    color = (90, 90, 90)
    if kind == "jeep":
        image = np.full((260, 340, 3), 35, np.uint8)
        cv2.rectangle(image, (60, 30), (280, 210), color, -1)
        cv2.rectangle(image, (90, 48), (250, 120), (20, 20, 20), -1)
        cv2.rectangle(image, (70, 150), (270, 190), (70, 70, 70), -1)
        cv2.circle(image, (100, 210), 22, (15, 15, 15), -1)
        cv2.circle(image, (240, 210), 22, (15, 15, 15), -1)
        px = plate_x or 130
        cv2.rectangle(image, (px, 158), (px + 70, 184), (230, 230, 230), -1)
    else:
        image = np.full((180, 460, 3), 35, np.uint8)
        cv2.ellipse(image, (230, 110), (180, 42), 0, 0, 360, color, -1)
        cv2.ellipse(image, (210, 78), (70, 28), 0, 180, 360, (20, 20, 20), -1)
        cv2.circle(image, (120, 140), 18, (15, 15, 15), -1)
        cv2.circle(image, (340, 140), 18, (15, 15, 15), -1)
        px = plate_x or 200
        cv2.rectangle(image, (px, 112), (px + 60, 132), (230, 230, 230), -1)
    if snow:
        height, width = image.shape[:2]
        cv2.circle(image, (width // 2, int(height * 0.46)), 16, (245, 245, 245), -1)
        cv2.ellipse(image, (int(width * 0.66), int(height * 0.42)), (26, 9), 15, 0, 360, (250, 250, 250), -1)
    if night:
        dark = (image.astype(np.float32) * 0.28).astype(np.uint8)
        image = dark
        h, w = image.shape[:2]
        cv2.circle(image, (int(w * 0.32), int(h * 0.62)), 8, (255, 255, 255), -1)
        cv2.circle(image, (int(w * 0.68), int(h * 0.62)), 8, (255, 255, 255), -1)
    return image


def test_vector_length_and_norm():
    rng = np.random.default_rng(0)
    vector = shape_vector(_crop(rng, True))
    assert vector.shape == (SHAPE_DIM,)
    assert SHAPE_DIM == SILHOUETTE_DIM + FACE_DIM
    assert abs(float(np.linalg.norm(vector[:SILHOUETTE_DIM])) - 1.0) < 1e-5
    assert abs(float(np.linalg.norm(vector[SILHOUETTE_DIM:])) - 1.0) < 1e-5


def test_border_noise_does_not_dominate():
    crops = [_crop(np.random.default_rng(seed), True) for seed in (1, 2, 3)]
    vectors = [shape_vector(crop) for crop in crops]
    sims = [score_gallery(vectors[i], vectors[j])
            for i in range(len(vectors)) for j in range(i + 1, len(vectors))]
    assert min(sims) > 0.90


def test_reference_bodies_stay_separate_in_night_and_snow():
    jeep = shape_vector(_vehicle("jeep"))
    jeep_night = shape_vector(_vehicle("jeep", night=True, plate_x=150))
    sport = shape_vector(_vehicle("sport"))
    sport_snow = shape_vector(_vehicle("sport", snow=True, plate_x=230))
    def silhouette(left, right) -> float:
        return float(l2_normalize(left[:SILHOUETTE_DIM]) @ l2_normalize(right[:SILHOUETTE_DIM]))

    assert silhouette(jeep, jeep_night) >= 0.90
    assert silhouette(sport, sport_snow) >= 0.90
    assert score_gallery(jeep, sport) <= 0.72
    jeep_gallery = select_prototypes(np.stack([jeep, jeep_night]))
    sport_gallery = select_prototypes(np.stack([sport, sport_snow]))
    assert score_gallery(jeep_night, jeep_gallery) >= 0.90
    assert score_gallery(sport_snow, sport_gallery) >= 0.90
    assert score_gallery(sport, jeep_gallery) <= 0.72
    assert score_gallery(jeep, sport_gallery) <= 0.72


def test_other_body_in_the_same_clip_is_not_averaged_in():
    jeep = shape_vector(_vehicle("jeep"))
    sport = shape_vector(_vehicle("sport"))
    mixed = np.stack([jeep, jeep, sport])
    kept = select_prototypes(mixed, min_sim=0.90)
    assert len(kept) == 2
    assert float(score_gallery(sport, kept)) < float(score_gallery(jeep, kept))


def test_distant_view_of_the_same_body_stays_in_the_gallery():
    """Дальний кадр того же силуэта остаётся в галерее, даже если передок не совпал."""
    close = shape_vector(_vehicle("jeep"))
    other_face = np.zeros(FACE_DIM, np.float32)
    other_face[::2] = 1.0
    distant = np.concatenate([close[:SILHOUETTE_DIM], l2_normalize(other_face)])
    close_only = select_prototypes(np.stack([close, close, close]))
    kept = select_prototypes(np.stack([close, close, distant, distant]))
    assert score_gallery(distant, close_only) < 0.80
    assert score_gallery(distant, kept) >= 0.90


def test_legacy_single_vector_still_unpacks():
    vector = shape_vector(_vehicle("jeep"))
    restored = unpack_gallery(vector.astype(np.float32).tobytes())
    assert restored.shape == (1, vector.shape[0])
    gallery = unpack_gallery(pack_gallery(np.stack([vector, vector])))
    assert gallery.shape[0] == 2
    assert abs(score_gallery(gallery[0], vector) - 1.0) < 1e-5


def test_thin_front_is_not_a_tall_grille():
    """Седан и кроссовер с одного ракурса: общий ящик похож, передок нет."""
    def draw(tall: bool, night: bool = False) -> np.ndarray:
        image = np.full((280, 380, 3), 40, np.uint8)
        cv2.rectangle(image, (36, 28), (344, 248), (110, 110, 110), -1)
        cv2.rectangle(image, (70, 46), (250, 118), (18, 18, 18), -1)
        if tall:
            for y in range(150, 214, 7):
                cv2.line(image, (188, y), (310, y), (25, 25, 25), 1)
            for x in range(188, 310, 7):
                cv2.line(image, (x, 150), (x, 214), (25, 25, 25), 1)
        else:
            cv2.line(image, (188, 178), (310, 178), (25, 25, 25), 3)
        cv2.circle(image, (90, 236), 24, (12, 12, 12), -1)
        cv2.circle(image, (270, 236), 24, (12, 12, 12), -1)
        if night:
            image = (image.astype(np.float32) * 0.28).astype(np.uint8)
        return image

    tall = shape_vector(draw(True))
    tall_night = shape_vector(draw(True, night=True))
    slit = shape_vector(draw(False))
    gallery = select_prototypes(np.stack([tall, tall_night]))
    assert score_gallery(tall_night, gallery) >= 0.85
    assert score_gallery(slit, gallery) < 0.80


def test_marked_crop_ignores_barrier():
    frame = np.full((200, 320, 3), 80, np.uint8)
    cv2.rectangle(frame, (20, 30), (140, 150), (0, 200, 255), 2)
    cv2.rectangle(frame, (10, 170), (300, 190), (0, 140, 255), -1)  # шлагбаум
    crop = crop_marked_vehicle(frame)
    assert crop is not None
    assert crop.shape[0] < 140
    assert crop.shape[1] < 150


def test_snapshot_name_cannot_leave_the_folder():
    assert safe_plate_token("../../secret") == "secret"
    assert safe_plate_token("А001АА799") == "А001АА799"
    assert ".." not in safe_plate_token("..\\windows\\system")
    assert safe_plate_token("") == "plate"
