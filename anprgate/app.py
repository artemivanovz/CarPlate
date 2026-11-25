"""Сборка компонентов приложения по конфигурации."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import AppConfig
from .db import Database
from .models import build_body_embedder, build_color_estimator, build_plate_reader, build_vehicle_detector
from .pipeline import Pipeline
from .video import VideoSource

log = logging.getLogger(__name__)


@dataclass
class Models:
    detector: object
    plate_reader: object
    color_estimator: object
    body_embedder: object = None


def load_models(cfg: AppConfig) -> Models:
    log.info("Загрузка моделей...")
    # FastALPR подготавливает общие библиотеки CUDA/cuDNN перед созданием ONNX-сессий.
    plate_reader = build_plate_reader(cfg)
    body_embedder = None
    if cfg.access.body_check:
        log.info("Загрузка сверки кузова...")
        body_embedder = build_body_embedder(cfg)
    return Models(
        detector=build_vehicle_detector(cfg),
        plate_reader=plate_reader,
        color_estimator=build_color_estimator(cfg),
        body_embedder=body_embedder,
    )


def build_pipeline(cfg: AppConfig, db: Database, models: Models, source: VideoSource) -> Pipeline:
    # Трекер Ultralytics хранит состояние внутри модели; новый источник — новый трек-контекст.
    predictor = getattr(models.detector.model, "predictor", None)
    if predictor is not None and hasattr(predictor, "trackers"):
        del predictor.trackers
    return Pipeline(
        cfg,
        db,
        models.detector,
        models.plate_reader,
        models.color_estimator,
        source_name=source.display_name,
        media_is_file=not source.is_stream,
        body_embedder=models.body_embedder,
    )
