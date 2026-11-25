"""Детектор и трекер ТС на Ultralytics YOLO (COCO) + ByteTrack."""

from __future__ import annotations

from typing import List

import numpy as np

from ..config import VehicleModelConfig
from .base import VehicleDetection


def resolve_torch_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class UltralyticsVehicleDetector:
    def __init__(self, cfg: VehicleModelConfig, weights_path: str) -> None:
        from ultralytics import YOLO

        self.cfg = cfg
        self.device = resolve_torch_device(cfg.device)
        self.model = YOLO(weights_path)
        self.names = self.model.names

    def track(self, frame: np.ndarray) -> List[VehicleDetection]:
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.cfg.tracker,
            classes=self.cfg.classes,
            conf=self.cfg.conf,
            imgsz=self.cfg.imgsz,
            device=self.device,
            verbose=False,
        )
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
        return [
            VehicleDetection(
                track_id=None if tid is None else int(tid),
                bbox=(int(b[0]), int(b[1]), int(b[2]), int(b[3])),
                label=str(self.names.get(int(c), c)),
                conf=float(cf),
            )
            for b, cf, c, tid in zip(xyxy, confs, classes, ids)
        ]
