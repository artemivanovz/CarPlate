"""Конвейер обработки кадра: ТС → номер → цвет → голосование → решение → журнал."""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .access import REVIEW, AccessDecision, BodyMatch, decide, find_vehicle
from .config import AppConfig
from .db import Database, EventRecord
from .models.base import BBox, ColorEstimate, ColorEstimator, PlateReader, VehicleDetector
from .models.body import score_gallery, unpack_gallery
from .models.color import scene_color_reason
from .plates import canonical, is_valid_ru
from .tracking import PlateVote, TrackBook, TrackState
from .video import Frame

log = logging.getLogger(__name__)

MAX_COLOR_SAMPLES = 15


@dataclass
class TrackOverlay:
    track_id: int
    bbox: BBox
    label: str
    plate: Optional[str]
    plate_bbox: Optional[BBox]
    color: Optional[str]
    votes: int
    status: str  # pending | allow | deny | review
    body_score: Optional[float] = None


@dataclass
class AccessEvent:
    event_id: int
    ts: datetime
    media_pos: Optional[float]
    source: str
    track_id: int
    plate: str
    plate_raw: str
    plate_conf: float
    plate_valid: bool
    votes: int
    color: ColorEstimate
    decision: AccessDecision
    frame: Optional[np.ndarray]
    plate_crop: Optional[np.ndarray]
    vehicle_crop: Optional[np.ndarray]
    snapshot_path: Optional[str]
    vehicle_plate_bbox: Optional[BBox] = None


@dataclass
class FrameResult:
    index: int
    ts: float
    overlays: List[TrackOverlay] = field(default_factory=list)
    events: List[AccessEvent] = field(default_factory=list)
    processed: bool = False
    infer_ms: float = 0.0


def _clip_box(box: BBox, w: int, h: int, margin: float = 0.0) -> BBox:
    x1, y1, x2, y2 = box
    mx, my = int((x2 - x1) * margin), int((y2 - y1) * margin)
    return max(0, x1 - mx), max(0, y1 - my), min(w, x2 + mx), min(h, y2 + my)


_PLATE_FILE = re.compile(r"[^0-9A-Za-zА-Яа-яЁё]+")


def safe_plate_token(plate: str) -> str:
    """Имя файла снимка не содержит путей и паролей — только буквы и цифры номера."""
    token = _PLATE_FILE.sub("", plate or "")
    return token[:16] or "plate"


def imwrite_unicode(path: Path, image: np.ndarray) -> None:
    """cv2.imwrite не поддерживает не-ASCII пути в Windows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(path.suffix or ".jpg", image)
    if ok:
        buf.tofile(str(path))


class Pipeline:
    def __init__(
        self,
        cfg: AppConfig,
        db: Database,
        detector: VehicleDetector,
        plate_reader: PlateReader,
        color_estimator: ColorEstimator,
        source_name: str,
        media_is_file: bool = True,
        body_embedder=None,
    ) -> None:
        self.cfg = cfg
        self.db = db
        self.detector = detector
        self.plate_reader = plate_reader
        self.color_estimator = color_estimator
        self.body_embedder = body_embedder
        self.source_name = source_name
        self.media_is_file = media_is_file
        self.book = TrackBook(cfg.processing)
        self._last_decided: Dict[str, float] = {}
        self._decided_tracks: Dict[int, str] = {}
        self._last_overlays: List[TrackOverlay] = []

    # --- Обработка кадра ------------------------------------------------

    def process(self, frame: Frame) -> FrameResult:
        result = FrameResult(frame.index, frame.ts)
        if frame.index % max(1, self.cfg.processing.frame_interval) != 0:
            result.overlays = self._last_overlays
            return result

        started = time.perf_counter()
        image = frame.image
        limit = self.cfg.processing.max_frame_side
        if limit and max(image.shape[:2]) > limit:
            scale = limit / max(image.shape[:2])
            image = cv2.resize(
                image,
                (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))),
                interpolation=cv2.INTER_AREA,
            )
        h, w = image.shape[:2]
        scene_reason = scene_color_reason(image, self.cfg.models.color)

        overlays: List[TrackOverlay] = []
        for det in self.detector.track(image):
            if det.track_id is None:
                continue
            state = self.book.touch(det.track_id, frame.ts, det.bbox, det.label)
            plate_bbox: Optional[BBox] = None
            if not state.decided:
                plate_bbox = self._observe(state, image, det.bbox, scene_reason, w, h)
                if self.book.ready(state):
                    event = self._finalize(state, frame.ts)
                    if event:
                        result.events.append(event)
            overlay = self._overlay(state, plate_bbox)
            if image.shape[:2] != frame.image.shape[:2]:
                sx, sy = frame.image.shape[1] / w, frame.image.shape[0] / h

                def source_box(box):
                    x1, y1, x2, y2 = box
                    return round(x1 * sx), round(y1 * sy), round(x2 * sx), round(y2 * sy)

                overlay = replace(
                    overlay,
                    bbox=source_box(overlay.bbox),
                    plate_bbox=source_box(overlay.plate_bbox) if overlay.plate_bbox else None,
                )
            overlays.append(overlay)

        for state in self.book.pop_lost(frame.ts):
            if not state.decided and state.votes:
                event = self._finalize(state, frame.ts)
                if event:
                    result.events.append(event)
            self._decided_tracks.pop(state.track_id, None)

        result.overlays = overlays
        result.processed = True
        result.infer_ms = (time.perf_counter() - started) * 1000.0
        self._last_overlays = overlays
        return result

    def flush(self, ts: float) -> List[AccessEvent]:
        """Принимает решения по незавершённым трекам (конец видеофайла)."""
        events = []
        for state in list(self.book.tracks.values()):
            if not state.decided and state.votes:
                event = self._finalize(state, ts)
                if event:
                    events.append(event)
        self.book.tracks.clear()
        return events

    def _observe(self, state: TrackState, image: np.ndarray, vbox: BBox,
                 scene_reason: Optional[str], w: int, h: int) -> Optional[BBox]:
        cx1, cy1, cx2, cy2 = _clip_box(vbox, w, h, margin=0.05)
        crop = image[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        readings = [r for r in self.plate_reader.read(crop) if r.text]
        best = max(readings, key=lambda r: r.det_conf * r.ocr_conf, default=None)
        if best is None or best.ocr_conf < self.cfg.processing.min_ocr_conf:
            return None

        vehicle_crop, scaled_plate = self._bounded_vehicle_crop(crop, best.bbox)
        if scene_reason:
            state.colors.append(ColorEstimate(None, None, scene_reason))
        else:
            state.colors.append(self.color_estimator.estimate(vehicle_crop, exclude=scaled_plate))
        del state.colors[:-MAX_COLOR_SAMPLES]

        px1, py1, px2, py2 = best.bbox
        plate_bbox = (px1 + cx1, py1 + cy1, px2 + cx1, py2 + cy1)
        plate = canonical(best.text)
        vote = PlateVote(plate=plate, raw=best.text, conf=best.ocr_conf, valid=is_valid_ru(plate))
        state.add_vote(vote, image, crop[py1:py2, px1:px2], vehicle_crop, vbox, plate_bbox=scaled_plate)
        return plate_bbox

    # --- Решение и журнал -------------------------------------------------

    def _finalize(self, state: TrackState, ts: float) -> Optional[AccessEvent]:
        state.decided = True
        lead = state.leader()
        if lead is None:
            return None
        vote, count, share = lead
        state.decided_plate = vote.plate

        last = self._last_decided.get(vote.plate)
        if last is not None and abs(ts - last) < self.cfg.processing.repeat_cooldown_sec:
            log.debug("Повтор номера %s в пределах интервала, событие не создаётся", vote.plate)
            state.decision = self._decided_tracks.get(state.track_id, "pending")
            return None
        self._last_decided[vote.plate] = ts

        color = state.color()
        state.decided_color = color.name
        vehicles = self.db.list_vehicles()
        body = self._body_match(state, vote.plate, vehicles) if self.cfg.access.body_check else None
        decision = decide(vote.plate, vote.valid, color, vehicles, self.cfg.access, body)
        state.body_score = decision.body_score
        if count < self.cfg.processing.min_votes:
            decision.escalate(REVIEW, f"мало согласованных прочтений номера: {count}")
        if share < self.cfg.processing.consensus:
            decision.escalate(REVIEW, f"недостаточное согласие по номеру: {share:.0%}")
        state.decision = decision.decision
        self._decided_tracks[state.track_id] = decision.decision

        now = datetime.now()
        snapshot_path, crop_path = self._save_snapshots(state, vote.plate, now)
        record = EventRecord(
            ts=now,
            source=self.source_name,
            track_id=state.track_id,
            plate_raw=vote.raw,
            plate=vote.plate,
            plate_conf=round(vote.conf, 4),
            plate_valid=vote.valid,
            votes=len(state.votes),
            color_detected=color.name,
            color_lab=color.lab,
            vehicle_id=decision.vehicle.id if decision.vehicle else None,
            match_distance=decision.match_distance,
            decision=decision.decision,
            reasons=decision.reasons,
            snapshot_path=snapshot_path,
            plate_crop_path=crop_path,
            media_pos=round(ts, 2) if self.media_is_file else None,
        )
        event_id = self.db.log_event(record)
        body_text = f", кузов={decision.body_score:.2f}" if decision.body_score is not None else ""
        log.info("Событие #%d: %s, цвет=%s%s → %s (%s)", event_id, vote.plate, color.name,
                 body_text, decision.decision, "; ".join(decision.reasons))
        return AccessEvent(
            event_id=event_id,
            ts=now,
            media_pos=record.media_pos,
            source=self.source_name,
            track_id=state.track_id,
            plate=vote.plate,
            plate_raw=vote.raw,
            plate_conf=vote.conf,
            plate_valid=vote.valid,
            votes=len(state.votes),
            color=color,
            decision=decision,
            frame=state.best_frame,
            plate_crop=state.best_plate_crop,
            vehicle_crop=state.best_vehicle_crop,
            snapshot_path=snapshot_path,
            vehicle_plate_bbox=state.best_vehicle_plate_bbox,
        )

    def _save_snapshots(self, state: TrackState, plate: str, now: datetime):
        if state.best_frame is None:
            return None, None
        folder = self.cfg.snapshots_dir / now.strftime("%Y%m%d")
        stem = f"{now.strftime('%H%M%S_%f')[:-3]}_t{state.track_id}_{safe_plate_token(plate)}"
        frame = state.best_frame.copy()
        x1, y1, x2, y2 = state.best_bbox or state.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        snapshot = (folder / f"{stem}.jpg").resolve()
        if self.cfg.snapshots_dir.resolve() not in snapshot.parents:
            log.error("Снимок отклонён: путь выходит за каталог snapshots")
            return None, None
        imwrite_unicode(snapshot, frame)
        crop_path = None
        if state.best_plate_crop is not None and state.best_plate_crop.size:
            crop_path = folder / f"{stem}_plate.jpg"
            imwrite_unicode(crop_path, state.best_plate_crop)
        return str(snapshot), (str(crop_path) if crop_path else None)

    def _overlay(self, state: TrackState, plate_bbox: Optional[BBox]) -> TrackOverlay:
        lead = state.leader()
        plate = state.decided_plate or (lead[0].plate if lead else None)
        color = state.decided_color if state.decided else state.color().name
        return TrackOverlay(
            track_id=state.track_id,
            bbox=state.bbox,
            label=state.label,
            plate=plate,
            plate_bbox=plate_bbox,
            color=color,
            votes=len(state.votes),
            status=state.decision or "pending",
            body_score=state.body_score,
        )

    def _body_match(self, state: TrackState, plate: str, vehicles) -> BodyMatch:
        vehicle, _distance = find_vehicle(plate, vehicles, self.cfg.access.fuzzy_max_distance)
        if vehicle is None or not vehicle.active:
            return BodyMatch(None, reason="нет эталона кузова")
        if self.body_embedder is None:
            return BodyMatch(None, reason="сверка кузова не выполнена")
        raw = self.db.body_vector(vehicle.id)
        if not raw:
            return BodyMatch(None, reason="нет эталона кузова")
        samples = state.body_samples
        if not samples and state.best_vehicle_crop is not None:
            samples = [(0.0, state.best_vehicle_crop, None)]
        if not samples:
            return BodyMatch(None, reason="нет кадра кузова")
        crops = [crop for _sharp, crop, _plate in samples]
        plates = [box for _sharp, _crop, box in samples]
        try:
            gallery = unpack_gallery(raw)
            vectors = self.body_embedder.embed(crops, plates)
            if len(vectors) == 0 or vectors.ndim != 2 or gallery.ndim != 2:
                return BodyMatch(None, reason="сверка кузова не выполнена")
            score = score_gallery(vectors, gallery)
        except Exception:
            log.exception("Сверка кузова не выполнена")
            return BodyMatch(None, reason="сверка кузова не выполнена")
        return BodyMatch(score, frames=len(vectors), references=int(gallery.shape[0]))

    def _bounded_vehicle_crop(self, crop: np.ndarray, plate_bbox: Optional[BBox]):
        """Кроп ТС, уменьшенный до предела конфига (для цвета, кузова и снапшотов)."""
        limit = self.cfg.processing.max_crop_side
        h, w = crop.shape[:2]
        if not limit or max(h, w) <= limit:
            return crop, plate_bbox
        scale = limit / max(h, w)
        small = cv2.resize(
            crop,
            (max(1, round(w * scale)), max(1, round(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
        if plate_bbox is None:
            return small, None
        x1, y1, x2, y2 = plate_bbox
        scaled = (round(x1 * scale), round(y1 * scale), round(x2 * scale), round(y2 * scale))
        return small, scaled
