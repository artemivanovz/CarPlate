import importlib
import json
import hashlib
import pickle

import pytest
import torch

from anprgate import assets
from anprgate import models as factory
from anprgate.config import AppConfig
from anprgate.models import body, fastreid_r50, ru_plate_reader, vehicle_detector


class UnapprovedMetadata:
    pass


@pytest.mark.parametrize("reader", ["fastreid", "ru_anpr"])
def test_checkpoint_with_python_objects_is_rejected_without_unsafe_retry(tmp_path, monkeypatch, reader):
    checkpoint = tmp_path / "untrusted.pt"
    torch.save({"metadata": UnapprovedMetadata()}, checkpoint)
    if reader == "fastreid":
        monkeypatch.setattr(fastreid_r50, "FastReIDSBS", lambda: pytest.fail("unverified checkpoint reached the model"))
        load = lambda: fastreid_r50.load_fastreid(str(checkpoint), "cpu")
    else:
        monkeypatch.syspath_prepend(str(ru_plate_reader.THIRD_PARTY / "ocr_model"))
        inference = importlib.import_module("nn.inference")
        monkeypatch.setattr(inference, "PlateOCR", lambda **kwargs: pytest.fail("unverified checkpoint reached the model"))
        load = lambda: inference.PlateRecognizer.from_checkpoint(str(checkpoint), device=torch.device("cpu"))
    with pytest.raises(pickle.UnpicklingError, match="Weights only load failed"):
        load()


@pytest.mark.parametrize("name", assets.STANDARD_MODELS)
def test_tampered_standard_weights_are_rejected_before_model_initialization(tmp_path, monkeypatch, name):
    original = b"verified model bytes"
    models = []
    for model_name in assets.STANDARD_MODELS:
        target = tmp_path / model_name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(original)
        models.append({"path": model_name, "size": len(original), "required": True,
                       "sha256": hashlib.sha256(original).hexdigest()})
    (tmp_path / "weights/manifest.json").write_text(json.dumps({"models": models}), encoding="utf-8")
    (tmp_path / name).write_bytes(b"modified checkpoint")
    monkeypatch.setattr(assets, "PROJECT", tmp_path)
    cfg = AppConfig()
    cfg.base_dir = tmp_path
    if name == "weights/yolo26n.pt":
        monkeypatch.setattr(vehicle_detector, "UltralyticsVehicleDetector", lambda *args: pytest.fail("unverified weights"))
        load = lambda: factory.build_vehicle_detector(cfg)
    elif name == assets.FASTREID_MODEL:
        cfg.models.body.backend = "fastreid"
        monkeypatch.setattr(body, "FastReIDBodyEmbedder", lambda *args: pytest.fail("unverified weights"))
        load = lambda: factory.build_body_embedder(cfg)
    else:
        cfg.models.plate.backend = "ru_anpr"
        monkeypatch.setattr(ru_plate_reader, "_ensure_rms_norm", lambda: pytest.fail("unverified weights"))
        load = lambda: factory.build_plate_reader(cfg)
    with pytest.raises(ValueError, match="SHA256"):
        load()
