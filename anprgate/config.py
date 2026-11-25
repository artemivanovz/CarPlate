"""Загрузка конфигурации из YAML в типизированные dataclass-объекты."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar

import yaml

T = TypeVar("T")

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "anpr.yaml"
EXAMPLE_CONFIG = DEFAULT_CONFIG.with_name("anpr.example.yaml")


@dataclass
class SourceConfig:
    uri: str = "data/videos/test.mp4"
    loop: bool = False
    realtime: bool = True


@dataclass
class PathsConfig:
    db: str = "data/anpr.sqlite3"
    snapshots: str = "data/snapshots"


@dataclass
class VehicleModelConfig:
    weights: str = "yolo26n.pt"
    device: str = "auto"
    conf: float = 0.35
    imgsz: int = 640
    classes: List[int] = field(default_factory=lambda: [2, 3, 5, 7])
    tracker: str = "bytetrack.yaml"


@dataclass
class PlateModelConfig:
    backend: str = "fast_alpr"
    detector_model: str = "yolo-v9-t-384-license-plate-end2end"
    ocr_model: str = "cct-s-v2-global-model"
    detector_conf: float = 0.4
    device: str = "auto"
    min_plate_width_px: int = 40
    weights_dir: str = "weights/ru_plate"


@dataclass
class ColorModelConfig:
    backend: str = "kmeans_lab"
    clusters: int = 4
    night_brightness: float = 45.0
    ir_channel_diff: float = 3.0
    dark_majority: float = 0.6


@dataclass
class BodyModelConfig:
    backend: str = "shape"          # shape — контур без цвета; fastreid — эмбеддинг VeRi, чувствителен к краске
    weights: str = "weights/reid/veri_sbs_R50-ibn.pth"
    device: str = "auto"
    input_size: List[int] = field(default_factory=lambda: [256, 256])
    plate_pad: float = 0.3


@dataclass
class ModelsConfig:
    vehicle: VehicleModelConfig = field(default_factory=VehicleModelConfig)
    plate: PlateModelConfig = field(default_factory=PlateModelConfig)
    color: ColorModelConfig = field(default_factory=ColorModelConfig)
    body: BodyModelConfig = field(default_factory=BodyModelConfig)


@dataclass
class ProcessingConfig:
    frame_interval: int = 2
    min_votes: int = 3
    max_votes: int = 12
    consensus: float = 0.5
    min_ocr_conf: float = 0.6
    lost_after_sec: float = 1.5
    repeat_cooldown_sec: float = 15.0
    max_frame_side: int = 1920
    max_crop_side: int = 640


@dataclass
class AccessConfig:
    require_valid_format: bool = True
    fuzzy_max_distance: int = 1
    fuzzy_decision: str = "review"
    color_check: bool = True
    color_unknown_decision: str = "review"
    color_delta_e_max: float = 28.0
    body_check: bool = False
    body_match: float = 0.75
    body_alarm: float = 0.45


@dataclass
class UIConfig:
    popup_autoclose_sec: float = 0.0
    max_events_in_list: int = 200


@dataclass
class AppConfig:
    source: SourceConfig = field(default_factory=SourceConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    models: ModelsConfig = field(default_factory=ModelsConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    access: AccessConfig = field(default_factory=AccessConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    base_dir: Path = field(default_factory=lambda: DEFAULT_CONFIG.parent)

    def resolve(self, relative: str) -> Path:
        path = Path(relative)
        return path if path.is_absolute() else (self.base_dir / path)

    @property
    def db_path(self) -> Path:
        return self.resolve(self.paths.db)

    @property
    def snapshots_dir(self) -> Path:
        return self.resolve(self.paths.snapshots)

    def source_uri(self) -> str:
        uri = self.source.uri
        if "://" in uri:
            return uri
        return str(self.resolve(uri))


def _build(cls: Type[T], data: Optional[Dict[str, Any]]) -> T:
    data = data or {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        raise ValueError(f"{cls.__name__}: неизвестные параметры {sorted(unknown)}")
    kwargs: Dict[str, Any] = {}
    for name, value in data.items():
        default = known[name].default_factory() if callable(known[name].default_factory) else None
        if is_dataclass(default):
            kwargs[name] = _build(type(default), value)
        else:
            kwargs[name] = value
    return cls(**kwargs)


def load_config(path: Optional[Path] = None) -> AppConfig:
    path = Path(path) if path else (DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else EXAMPLE_CONFIG)
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    cfg = _build(AppConfig, raw)
    cfg.base_dir = path.resolve().parent
    env_source = os.environ.get("ANPR_SOURCE")
    if env_source:
        cfg.source.uri = env_source
    return cfg
