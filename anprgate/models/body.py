"""Эмбеддинг кузова. Номер не входит в вектор: у shape-контура область таблички обнуляется.

У каждой записи в базе своя галерея векторов. Чужой кузов в эту галерею не усредняется.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import cv2
import numpy as np

from ..config import BodyModelConfig
from .base import BBox
from .fastreid_r50 import fastreid_embed_fn
from .vehicle_detector import resolve_torch_device


def mask_plate(crop: np.ndarray, plate: Optional[BBox], pad: float) -> np.ndarray:
    """Закрашивает номер медианой кузова, чтобы сеть не узнавала машину по табличке."""
    if plate is None or crop.size == 0:
        return crop
    h, w = crop.shape[:2]
    x1, y1, x2, y2 = plate
    mx, my = int((x2 - x1) * pad), int((y2 - y1) * pad)
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)
    if x2 <= x1 or y2 <= y1:
        return crop
    out = crop.copy()
    med = np.median(crop.reshape(-1, crop.shape[-1]), axis=0).astype(crop.dtype)
    out[y1:y2, x1:x2] = med
    return out


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + 1e-8)


_CANVAS = 128
_CELL = 16
_BINS = 8
_ASPECT_WEIGHT = 0.45
_FACE_CANVAS = 64
_FACE_CELL = 8
# Передок сравниваем только на тесном кадре спереди (кроссовер и седан ~1.3–1.4).
# Кроп купе шире 1.6: окно попадает мимо решётки, и эта проверка его не заваливает.
_FACE_MAX_ASPECT = 1.60
# Силуэт всего кропа и отдельно передок (решётка и оптика).
# Общий контур двух Чери с одного ракурса даёт 0.86–0.91 и не отличает седан от кроссовера.
SILHOUETTE_DIM = ((_CANVAS // _CELL - 1) ** 2) * (4 * _BINS) + 16
FACE_DIM = ((_FACE_CANVAS // _FACE_CELL - 1) ** 2) * (4 * _BINS)
SHAPE_DIM = SILHOUETTE_DIM + FACE_DIM
_GALLERY_MAGIC = b"BDY1"


def _content_view(bgr: np.ndarray):
    """Вписывает кадр в квадрат без растяжения. Поля не получают градиента."""
    height, width = bgr.shape[:2]
    scale = _CANVAS / float(max(height, width))
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    resized = cv2.resize(bgr, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    origin = ((_CANVAS - resized_w) // 2, (_CANVAS - resized_h) // 2)
    return resized, scale, origin


def shape_parts(bgr: np.ndarray, plate: Optional[BBox] = None, plate_pad: float = 0.0):
    """Сетка градиентов и пропорция кадра. Обе части с нормой 1."""
    resized, scale, (x0, y0) = _content_view(bgr)
    # Максимум каналов, а не яркость: красная и серая краска одной формы остаются контуром.
    gray = resized.max(axis=2)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    content_h, content_w = gray.shape
    inset = max(1, int(round(0.04 * min(content_h, content_w))))
    region = np.zeros((content_h, content_w), np.float32)
    if content_h > 2 * inset and content_w > 2 * inset:
        region[inset:content_h - inset, inset:content_w - inset] = 1.0
    if plate is not None:
        px1, py1, px2, py2 = plate
        grow_x = int(abs(px2 - px1) * plate_pad)
        grow_y = int(abs(py2 - py1) * plate_pad)
        rx1 = max(0, int(np.floor((min(px1, px2) - grow_x) * scale)))
        ry1 = max(0, int(np.floor((min(py1, py2) - grow_y) * scale)))
        rx2 = min(content_w, int(np.ceil((max(px1, px2) + grow_x) * scale)))
        ry2 = min(content_h, int(np.ceil((max(py1, py2) + grow_y) * scale)))
        if rx2 > rx1 and ry2 > ry1:
            region[ry1:ry2, rx1:rx2] = 0.0
    active = region > 0
    if np.any(active):
        # Маленькие почти белые пятна — фары. Белая краска крупнее и остаётся в контуре.
        lamps = ((gray >= 235) & active).astype(np.uint8)
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(lamps, connectivity=8)
        limit = 0.02 * float(gray.size)
        for index in range(1, count):
            if int(stats[index, cv2.CC_STAT_AREA]) <= limit:
                region[labels == index] = 0.0
        active = region > 0
    level = float(np.percentile(mag[active], 90)) + 1e-6 if np.any(active) else 1.0
    weight = np.clip(mag / level, 0.0, 1.0) * region

    canvas_w = np.zeros((_CANVAS, _CANVAS), np.float32)
    canvas_bin = np.zeros((_CANVAS, _CANVAS), np.int32)
    bins = np.floor((ang % 180.0) * (_BINS / 180.0)).astype(np.int32) % _BINS
    canvas_w[y0:y0 + content_h, x0:x0 + content_w] = weight
    canvas_bin[y0:y0 + content_h, x0:x0 + content_w] = bins
    cells = _CANVAS // _CELL
    hist = np.zeros((cells, cells, _BINS), np.float32)
    ys, xs = np.indices(canvas_w.shape)
    np.add.at(hist, (ys // _CELL, xs // _CELL, canvas_bin), canvas_w)
    blocks = []
    for y in range(cells - 1):
        for x in range(cells - 1):
            block = hist[y:y + 2, x:x + 2].ravel()
            blocks.append(block / (np.linalg.norm(block) + 1e-6))
    hog = np.concatenate(blocks).astype(np.float32)
    hog /= np.linalg.norm(hog) + 1e-6
    src_h, src_w = bgr.shape[:2]
    ratio = src_w / max(src_h, 1)
    centers = np.linspace(1.05, 2.4, 16).astype(np.float32)
    aspect = np.exp(-0.5 * ((centers - ratio) / 0.18) ** 2).astype(np.float32)
    aspect /= np.linalg.norm(aspect) + 1e-6
    return hog, aspect


def _face_vector(bgr: np.ndarray, plate: Optional[BBox] = None, plate_pad: float = 0.0) -> np.ndarray:
    """Мелкий контур только передка: высокая решётка кроссовера против узкой щели седана.

    Силуэт всего кадра у этих двух машин почти один и тот же, потому что камера и ракурс
    совпадают. Передок в общий HOG тонет. Номер и мелкие блики фар в эту сетку не входят.
    """
    height, width = bgr.shape[:2]
    y0, y1 = int(height * 0.52), int(height * 0.86)
    x0, x1 = int(width * 0.30), int(width * 0.88)
    if y1 - y0 < 8 or x1 - x0 < 8:
        return np.zeros(FACE_DIM, np.float32)
    crop = bgr[y0:y1, x0:x1]
    scale = _FACE_CANVAS / float(max(crop.shape[:2]))
    resized_w = max(1, int(round(crop.shape[1] * scale)))
    resized_h = max(1, int(round(crop.shape[0] * scale)))
    resized = cv2.resize(crop, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    raw = resized.max(axis=2)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(raw)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    region = np.ones(gray.shape, np.float32)
    if plate is not None:
        px1, py1, px2, py2 = plate
        grow_x = int(abs(px2 - px1) * plate_pad)
        grow_y = int(abs(py2 - py1) * plate_pad)
        rx1 = int(np.floor((min(px1, px2) - grow_x - x0) * scale))
        ry1 = int(np.floor((min(py1, py2) - grow_y - y0) * scale))
        rx2 = int(np.ceil((max(px1, px2) + grow_x - x0) * scale))
        ry2 = int(np.ceil((max(py1, py2) + grow_y - y0) * scale))
        rx1, ry1 = max(0, rx1), max(0, ry1)
        rx2, ry2 = min(resized_w, rx2), min(resized_h, ry2)
        if rx2 > rx1 and ry2 > ry1:
            region[ry1:ry2, rx1:rx2] = 0.0
    # До CLAHE: ночная фара остаётся белым пятном и не должна перебивать решётку.
    lamps = ((raw >= 235) & (region > 0)).astype(np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(lamps, connectivity=8)
    limit = 0.12 * float(raw.size)
    for index in range(1, count):
        if int(stats[index, cv2.CC_STAT_AREA]) <= limit:
            region[labels == index] = 0.0
    level = float(np.percentile(mag[region > 0], 90)) + 1e-6 if np.any(region > 0) else 1.0
    weight = np.clip(mag / level, 0.0, 1.0) * region
    canvas_w = np.zeros((_FACE_CANVAS, _FACE_CANVAS), np.float32)
    canvas_bin = np.zeros((_FACE_CANVAS, _FACE_CANVAS), np.int32)
    oy = (_FACE_CANVAS - resized_h) // 2
    ox = (_FACE_CANVAS - resized_w) // 2
    canvas_w[oy:oy + resized_h, ox:ox + resized_w] = weight
    bins = np.floor((ang % 180.0) * (_BINS / 180.0)).astype(np.int32) % _BINS
    canvas_bin[oy:oy + resized_h, ox:ox + resized_w] = bins
    cells = _FACE_CANVAS // _FACE_CELL
    hist = np.zeros((cells, cells, _BINS), np.float32)
    ys, xs = np.indices(canvas_w.shape)
    np.add.at(hist, (ys // _FACE_CELL, xs // _FACE_CELL, canvas_bin), canvas_w)
    blocks = []
    for y in range(cells - 1):
        for x in range(cells - 1):
            block = hist[y:y + 2, x:x + 2].ravel()
            blocks.append(block / (np.linalg.norm(block) + 1e-6))
    face = np.concatenate(blocks).astype(np.float32)
    return l2_normalize(face)


def shape_vector(bgr: np.ndarray, plate: Optional[BBox] = None, plate_pad: float = 0.0) -> np.ndarray:
    """Силуэт кузова и отдельно передок. Обе части уже единичной длины.

    Кадр не растягивается в квадрат. Общий контур не затирает передок: сверка
    берёт худшую из двух частей, поэтому похожий снаружи седан не проходит
    за кроссовер.
    """
    hog, aspect = shape_parts(bgr, plate, plate_pad)
    silhouette = l2_normalize(np.concatenate([hog, aspect * _ASPECT_WEIGHT]).astype(np.float32))
    ratio = bgr.shape[1] / max(bgr.shape[0], 1)
    if ratio >= _FACE_MAX_ASPECT:
        face = np.zeros(FACE_DIM, np.float32)
    else:
        face = _face_vector(bgr, plate, plate_pad)
    return np.concatenate([silhouette, face]).astype(np.float32)


def select_prototypes(vectors: np.ndarray, limit: int = 8, min_sim: float = 0.80) -> np.ndarray:
    """Оставляет кадры одного кузова. Второй кузов в том же ролике в эталон не входит."""
    data = np.asarray(vectors, dtype=np.float32)
    if data.size == 0:
        return np.empty((0, 0), np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    data = l2_normalize(data)
    if len(data) == 1:
        return data
    # В галерею берём силуэт, а не передок. Одна и та же машина в начале ролика
    # (ещё далеко) и вблизи даёт передок 0.49, и порог 0.80 выбрасывал дальний кадр.
    # Окно решает по первым кадрам, поэтому эталонный кроссовер получал 0.57.
    if data.shape[1] == SHAPE_DIM:
        sil = l2_normalize(data[:, :SILHOUETTE_DIM])
        sims = sil @ sil.T
    else:
        sims = data @ data.T
    center = int(np.argmax((sims >= min_sim).sum(axis=1)))
    members = np.flatnonzero(sims[center] >= min_sim)
    pool = data[members]
    if len(pool) <= limit:
        return pool
    chosen = [int(np.argmax(pool @ data[center]))]
    while len(chosen) < limit:
        nearest = (pool @ pool[chosen].T).max(axis=1)
        nxt = int(np.argmin(nearest))
        if nxt in chosen:
            break
        chosen.append(nxt)
    return pool[np.asarray(chosen, dtype=np.int32)]


def pack_gallery(vectors: np.ndarray) -> bytes:
    """Галерея одной машины. Формат не смешивается с вектором другой записи."""
    data = np.asarray(vectors, dtype=np.float32)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    _validate_gallery(data)
    data = np.ascontiguousarray(l2_normalize(data))
    count, dim = data.shape
    if count == 0 or dim == 0:
        raise ValueError("пустая галерея кузова")
    if count > 64:
        data = data[:64]
        count = 64
    header = _GALLERY_MAGIC + np.array([count, dim], dtype="<i4").tobytes()
    return header + data.tobytes()


def unpack_gallery(raw: bytes) -> np.ndarray:
    """Читает галерею BDY1 или прежний один вектор float32."""
    blob = bytes(raw)
    if blob.startswith(_GALLERY_MAGIC):
        if len(blob) < 12:
            raise ValueError("повреждён заголовок галереи кузова")
        count, dim = (int(v) for v in np.frombuffer(blob[4:12], dtype="<i4"))
        payload = blob[12:]
        if not (0 < count <= 64 and dim > 0 and len(payload) == count * dim * 4):
            raise ValueError("повреждён размер галереи кузова")
        data = np.frombuffer(payload, dtype="<f4").copy().reshape(count, dim)
    elif not blob:
        return np.empty((0, 0), np.float32)
    else:
        if len(blob) % 4:
            raise ValueError("повреждён вектор кузова")
        data = np.frombuffer(blob, dtype="<f4").copy().reshape(1, -1)
    _validate_gallery(data)
    return l2_normalize(data)


def _validate_gallery(data: np.ndarray) -> None:
    if data.ndim != 2 or not all(data.shape) or not np.isfinite(data).all():
        raise ValueError("некорректные значения галереи кузова")
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.linalg.norm(data, axis=1)
    if not np.isfinite(norms).all() or np.any(norms <= 1e-8):
        raise ValueError("пустой или некорректный вектор кузова")


def make_body_reference(embedder, crop: np.ndarray, plate_bbox: Optional[BBox]) -> bytes:
    """Эталон строится той же моделью и с той же маской номера, что и запрос."""
    if crop is None or crop.size == 0:
        raise ValueError("нет кадра кузова")
    if plate_bbox is None:
        raise ValueError("нет рамки номера для сохранения эталона")
    vectors = embedder.embed([crop], [plate_bbox])
    return pack_gallery(vectors)


def _pair_matrix(queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Попарная близость. Для полного вектора это худшая из силуэта и передка.

    Среднее их смешивает: седан с тем же ракурсом набирал 0.86 по силуэту и проходил.
    """
    if queries.shape[1] == SHAPE_DIM and gallery.shape[1] == SHAPE_DIM:
        q_sil = l2_normalize(queries[:, :SILHOUETTE_DIM])
        g_sil = l2_normalize(gallery[:, :SILHOUETTE_DIM])
        q_raw = queries[:, SILHOUETTE_DIM:]
        g_raw = gallery[:, SILHOUETTE_DIM:]
        usable = (np.linalg.norm(q_raw, axis=1)[:, None] > 0.5) & (
            np.linalg.norm(g_raw, axis=1)[None, :] > 0.5
        )
        face = l2_normalize(q_raw) @ l2_normalize(g_raw).T
        sil = q_sil @ g_sil.T
        return np.where(usable, np.minimum(sil, face), sil)
    return l2_normalize(queries) @ l2_normalize(gallery).T


def score_gallery(queries: np.ndarray, gallery: np.ndarray) -> float:
    """Близость кадров только к галерее этой машины. 75-й перцентиль, чтобы один шумный кадр не решал."""
    queries = np.asarray(queries, dtype=np.float32)
    gallery = np.asarray(gallery, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)
    if gallery.ndim == 1:
        gallery = gallery.reshape(1, -1)
    _validate_gallery(queries)
    _validate_gallery(gallery)
    if queries.shape[1] != gallery.shape[1] or gallery.shape[0] == 0:
        raise ValueError("разная длина вектора кузова")
    per_frame = _pair_matrix(queries, gallery).max(axis=1)
    if per_frame.size >= 3:
        return float(np.percentile(per_frame, 75))
    return float(np.max(per_frame))


def crop_marked_vehicle(frame: np.ndarray) -> Optional[np.ndarray]:
    """Кроп внутри жёлтой рамки события. Шлагбаум и другие оранжевые объекты в кроп не входят."""
    if frame.size == 0:
        return None
    blue, green, red = cv2.split(frame)
    # Рамка конвейера — BGR (0, 200, 255). Шлагбаум краснее и толще, его отсекаем по форме.
    mark = (
        (red > 220) & (green > 160) & (green < 235) & (blue < 55)
        & ((red.astype(np.int16) - green.astype(np.int16)) < 90)
    )
    mask = mark.astype(np.uint8) * 255
    if int(mask.sum()) < 40 * 255:
        return None
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    best = None
    best_area = 0
    for index in range(1, count):
        x, y, width, height, area = (int(v) for v in stats[index])
        if width < 24 or height < 24 or area < 40:
            continue
        component = labels[y:y + height, x:x + width] == index
        ys, xs = np.nonzero(component)
        if xs.size < 40:
            continue
        on_border = (xs <= 3) | (xs >= width - 4) | (ys <= 3) | (ys >= height - 4)
        # Контур прямоугольника лежит на границе своего бокса, диагональ шлагбаума — нет.
        if float(on_border.mean()) < 0.85:
            continue
        box_area = width * height
        if box_area > best_area:
            best_area = box_area
            best = (x, y, width, height)
    if best is None:
        return None
    x, y, width, height = best
    x1, y1 = x + 3, y + 3
    x2, y2 = x + width - 2, y + height - 2
    if x2 - x1 < 24 or y2 - y1 < 24:
        return None
    return frame[y1:y2, x1:x2].copy()


class FastReIDBodyEmbedder:
    """FastReID SBS R50-IBN, обученная на VeRi-776. Выход — 2048-d, L2-нормирован."""

    def __init__(self, cfg: BodyModelConfig, weights_path: str) -> None:
        self.cfg = cfg
        self.device = resolve_torch_device(cfg.device)
        self._fn = fastreid_embed_fn(weights_path, self.device, tuple(cfg.input_size))

    def embed(self, crops: Sequence[np.ndarray], plates: Sequence[Optional[BBox]]) -> np.ndarray:
        masked: List[np.ndarray] = []
        for crop, plate in zip(crops, plates):
            if crop is None or crop.size == 0 or min(crop.shape[:2]) < 24:
                continue
            rgb = cv2.cvtColor(mask_plate(crop, plate, self.cfg.plate_pad), cv2.COLOR_BGR2RGB)
            masked.append(rgb)
        if not masked:
            return np.empty((0, 0), dtype=np.float32)
        vectors = np.asarray(self._fn(masked), dtype=np.float32)
        return l2_normalize(vectors)


class ShapeBodyEmbedder:
    """Сверка кузова по контуру. Краска и время суток в вектор не входят."""

    device = "cpu"

    def __init__(self, cfg: BodyModelConfig) -> None:
        self.cfg = cfg

    def embed(self, crops: Sequence[np.ndarray], plates: Sequence[Optional[BBox]]) -> np.ndarray:
        vectors: List[np.ndarray] = []
        for crop, plate in zip(crops, plates):
            if crop is None or crop.size == 0 or min(crop.shape[:2]) < 24:
                continue
            vectors.append(shape_vector(crop, plate, self.cfg.plate_pad))
        if not vectors:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack(vectors)
