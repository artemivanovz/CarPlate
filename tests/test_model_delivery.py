import hashlib
import io
import json

import pytest

from anprgate import assets
from anprgate import models as factory
from anprgate.config import AppConfig


@pytest.fixture
def model(tmp_path):
    content = b"a verified model download"
    name = assets.FASTREID_MODEL
    (tmp_path / "weights").mkdir()
    metadata = {"path": name, "required": False, "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
                "url": "https://example.invalid/model.pth"}
    (tmp_path / "weights/manifest.json").write_text(json.dumps({"models": [metadata]}), encoding="utf-8")
    return tmp_path, name, content


def test_download_is_verified_and_then_reused_without_network(model, monkeypatch):
    root, name, content = model
    requests = []

    def download(request, **kwargs):
        requests.append(request.full_url)
        return io.BytesIO(content)

    monkeypatch.setattr(assets.urllib.request, "urlopen", download)
    target = assets.ensure_model(name, root)
    assert target.read_bytes() == content
    assert assets.ensure_model(name, root) == target
    assert requests == ["https://example.invalid/model.pth"]
    assert not list(target.parent.glob("*.part"))


@pytest.mark.parametrize("downloaded", [b"truncated", b"too much data" * 100, b"x" * 25])
def test_bad_download_is_not_published_and_partial_file_is_removed(model, monkeypatch, downloaded):
    root, name, _content = model
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *a, **kw: io.BytesIO(downloaded))
    with pytest.raises(ValueError):
        assets.ensure_model(name, root)
    assert not (root / name).exists()
    assert not list((root / name).parent.glob("*.part"))


def test_existing_modified_model_is_preserved(model, monkeypatch):
    root, name, _content = model
    target = root / name
    target.parent.mkdir(parents=True)
    target.write_bytes(b"a custom model")
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("unexpected download"))
    with pytest.raises(ValueError, match="not overwritten"):
        assets.ensure_model(name, root)
    assert target.read_bytes() == b"a custom model"


def test_missing_bundled_model_requests_git_restore_instead_of_network(tmp_path, monkeypatch):
    (tmp_path / "weights").mkdir()
    name = "weights/yolo26n.pt"
    metadata = {"path": name, "required": True, "size": 1, "sha256": "0" * 64}
    (tmp_path / "weights/manifest.json").write_text(json.dumps({"models": [metadata]}), encoding="utf-8")
    monkeypatch.setattr(assets.urllib.request, "urlopen", lambda *a, **kw: pytest.fail("unexpected download"))
    with pytest.raises(FileNotFoundError, match="git restore"):
        assets.ensure_model(name, tmp_path)


def test_download_path_cannot_leave_project(model):
    root, _name, _content = model
    with pytest.raises(ValueError, match="inside the project"):
        assets.ensure_model("../outside.pth", root)


def test_selecting_fastreid_prepares_its_missing_standard_model(tmp_path, monkeypatch):
    from anprgate.models import body

    cfg = AppConfig()
    cfg.base_dir = tmp_path
    cfg.models.body.backend = "fastreid"
    target = tmp_path / assets.FASTREID_MODEL
    prepared = []

    def prepare(name):
        prepared.append(name)
        target.parent.mkdir(parents=True)
        target.write_bytes(b"model")
        return target

    monkeypatch.setattr(assets, "PROJECT", tmp_path)
    monkeypatch.setattr(assets, "ensure_model", prepare)
    monkeypatch.setattr(body, "FastReIDBodyEmbedder", lambda config, path: (config, path))
    assert factory.build_body_embedder(cfg) == (cfg.models.body, str(target))
    assert prepared == [assets.FASTREID_MODEL]
