"""Фабрика нейросетевых модулей по конфигурации."""

from __future__ import annotations

from ..config import AppConfig
from .base import ColorEstimator, PlateReader, VehicleDetector


def build_vehicle_detector(cfg: AppConfig) -> VehicleDetector:
    from ..assets import verified_model_path
    from .vehicle_detector import UltralyticsVehicleDetector

    local = verified_model_path(cfg.resolve(cfg.models.vehicle.weights))
    return UltralyticsVehicleDetector(cfg.models.vehicle, str(local) if local.exists() else local.name)


def build_plate_reader(cfg: AppConfig) -> PlateReader:
    backend = cfg.models.plate.backend
    if backend == "fast_alpr":
        from .plate_reader import FastAlprPlateReader

        return FastAlprPlateReader(cfg.models.plate)
    if backend == "ru_anpr":
        from .ru_plate_reader import RuPlateReader

        return RuPlateReader(cfg.models.plate, cfg.resolve(cfg.models.plate.weights_dir))
    raise ValueError(f"Неизвестный backend распознавания номеров: {backend}")


def build_body_embedder(cfg: AppConfig):
    backend = cfg.models.body.backend
    if backend == "shape":
        from .body import ShapeBodyEmbedder

        return ShapeBodyEmbedder(cfg.models.body)
    if backend == "fastreid":
        from ..assets import verified_model_path
        from .body import FastReIDBodyEmbedder

        weights = verified_model_path(cfg.resolve(cfg.models.body.weights))
        if not weights.is_file():
            raise FileNotFoundError(f"Нет пользовательских весов сверки кузова: {weights}")
        return FastReIDBodyEmbedder(cfg.models.body, str(weights))
    raise ValueError(f"Неизвестный backend сверки кузова: {backend}")


def build_color_estimator(cfg: AppConfig) -> ColorEstimator:
    backend = cfg.models.color.backend
    if backend == "kmeans_lab":
        from .color import KMeansLabColorEstimator

        return KMeansLabColorEstimator(cfg.models.color)
    raise ValueError(f"Неизвестный backend определения цвета: {backend}")
