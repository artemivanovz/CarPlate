"""Оценка точности распознавания номеров на размеченных изображениях."""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from .models.base import PlateReader
from .plates import canonical, levenshtein


@dataclass
class Sample:
    image: Path
    plate: str
    group: str


@dataclass
class Outcome:
    sample: Sample
    predicted: Optional[str]
    ms: float

    @property
    def exact(self) -> bool:
        return self.predicted == self.sample.plate

    @property
    def detected(self) -> bool:
        return self.predicted is not None


def load_labels(path: Path, images_root: Path, delimiter: str = ";") -> List[Sample]:
    """CSV с колонками image, plate_num и необязательной plate_type (формат volga-anpr)."""
    samples: Dict[Path, Sample] = {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh, delimiter=delimiter):
            image = images_root / row["image"]
            if image.exists() and image not in samples:
                samples[image] = Sample(image, canonical(row["plate_num"]), row.get("plate_type") or "all")
    return list(samples.values())


def _imread(path: Path) -> Optional[np.ndarray]:
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def evaluate(reader: PlateReader, samples: List[Sample]) -> List[Outcome]:
    outcomes = []
    for sample in samples:
        image = _imread(sample.image)
        started = time.perf_counter()
        readings = reader.read(image) if image is not None else []
        ms = (time.perf_counter() - started) * 1000.0
        if readings:
            # В кадре оценки один целевой номер: берём ближайшее к разметке среди найденных.
            predicted = min((canonical(r.text) for r in readings), key=lambda p: levenshtein(p, sample.plate))
        else:
            predicted = None
        outcomes.append(Outcome(sample, predicted, ms))
    return outcomes


def summarize(outcomes: List[Outcome]) -> List[Dict[str, object]]:
    groups: Dict[str, List[Outcome]] = defaultdict(list)
    for o in outcomes:
        groups[o.sample.group].append(o)
    groups["ВСЕГО"] = outcomes
    rows = []
    for name, items in groups.items():
        n = len(items)
        rows.append({
            "group": name,
            "n": n,
            "detected": sum(o.detected for o in items) / n,
            "exact": sum(o.exact for o in items) / n,
            "within1": sum(o.detected and levenshtein(o.predicted, o.sample.plate) <= 1 for o in items) / n,
            "ms": float(np.median([o.ms for o in items])),
        })
    return rows


def write_errors(outcomes: List[Outcome], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.writer(fh, delimiter=";")
        writer.writerow(["image", "group", "expected", "predicted"])
        for o in outcomes:
            if not o.exact:
                writer.writerow([o.sample.image.name, o.sample.group, o.sample.plate, o.predicted or ""])
