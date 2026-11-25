"""Накопление прочтений номера и цвета по трекам ТС и голосование."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from .config import ProcessingConfig
from .models.base import BBox, ColorEstimate


@dataclass
class PlateVote:
    plate: str
    raw: str
    conf: float
    valid: bool


@dataclass
class TrackState:
    track_id: int
    first_seen: float
    last_seen: float
    bbox: BBox
    label: str
    votes: List[PlateVote] = field(default_factory=list)
    colors: List[ColorEstimate] = field(default_factory=list)
    best_conf: float = -1.0
    best_bbox: Optional[BBox] = None
    best_frame: Optional[np.ndarray] = None
    best_plate_crop: Optional[np.ndarray] = None
    best_vehicle_crop: Optional[np.ndarray] = None
    best_vehicle_plate_bbox: Optional[BBox] = None
    body_samples: List[Tuple[float, np.ndarray, Optional[BBox]]] = field(default_factory=list)
    body_score: Optional[float] = None
    decided: bool = False
    decided_plate: Optional[str] = None
    decided_color: Optional[str] = None
    decision: Optional[str] = None

    def add_vote(self, vote: PlateVote, frame: np.ndarray, plate_crop: np.ndarray,
                 vehicle_crop: np.ndarray, vehicle_bbox: BBox,
                 plate_bbox: Optional[BBox] = None) -> None:
        self.votes.append(vote)
        self._keep_body_sample(vehicle_crop, plate_bbox)
        if vote.conf > self.best_conf:
            self.best_conf = vote.conf
            self.best_bbox = vehicle_bbox
            self.best_frame = frame.copy()
            self.best_plate_crop = plate_crop.copy()
            self.best_vehicle_crop = vehicle_crop.copy()
            self.best_vehicle_plate_bbox = plate_bbox

    def _keep_body_sample(self, crop: np.ndarray, plate_bbox: Optional[BBox], limit: int = 4) -> None:
        if crop is None or crop.size == 0 or min(crop.shape[:2]) < 24:
            return
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        self.body_samples.append((sharp, crop.copy(), plate_bbox))
        self.body_samples.sort(key=lambda item: item[0], reverse=True)
        del self.body_samples[limit:]

    def leader(self) -> Optional[Tuple[PlateVote, int, float]]:
        """Лидирующий номер, число голосов за него и доля от всех голосов.

        Голос взвешивается уверенностью OCR; корректный формат РФ имеет приоритет.
        """
        if not self.votes:
            return None
        weights: Dict[str, float] = defaultdict(float)
        counts: Dict[str, int] = defaultdict(int)
        best_vote: Dict[str, PlateVote] = {}
        for vote in self.votes:
            weights[vote.plate] += max(vote.conf, 0.05) * (1.5 if vote.valid else 1.0)
            counts[vote.plate] += 1
            if vote.plate not in best_vote or vote.conf > best_vote[vote.plate].conf:
                best_vote[vote.plate] = vote
        plate = max(weights, key=weights.get)
        return best_vote[plate], counts[plate], counts[plate] / len(self.votes)

    def color(self) -> ColorEstimate:
        known = [c for c in self.colors if c.name is not None and c.lab is not None]
        if not known:
            reason = self.colors[-1].reason if self.colors else "нет кадров кузова"
            return ColorEstimate(None, None, reason)
        names: Dict[str, int] = defaultdict(int)
        for c in known:
            names[c.name] += 1
        name = max(names, key=names.get)
        labs = np.array([c.lab for c in known if c.name == name], dtype=np.float32)
        median = np.median(labs, axis=0)
        return ColorEstimate(name, (float(median[0]), float(median[1]), float(median[2])))


class TrackBook:
    def __init__(self, cfg: ProcessingConfig) -> None:
        self.cfg = cfg
        self.tracks: Dict[int, TrackState] = {}

    def touch(self, track_id: int, now: float, bbox: BBox, label: str) -> TrackState:
        state = self.tracks.get(track_id)
        if state is None:
            state = TrackState(track_id, now, now, bbox, label)
            self.tracks[track_id] = state
        state.last_seen, state.bbox, state.label = now, bbox, label
        return state

    def ready(self, state: TrackState) -> bool:
        if state.decided:
            return False
        return self.leader_ok(state) or len(state.votes) >= self.cfg.max_votes

    def leader_ok(self, state: TrackState) -> bool:
        lead = state.leader()
        return bool(lead) and lead[1] >= self.cfg.min_votes and lead[2] >= self.cfg.consensus

    def pop_lost(self, now: float) -> List[TrackState]:
        lost = [s for s in self.tracks.values() if now - s.last_seen > self.cfg.lost_after_sec]
        for state in lost:
            del self.tracks[state.track_id]
        return lost
