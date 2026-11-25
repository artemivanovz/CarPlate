"""Verified delivery of the optional model that exceeds GitHub's Git file limit."""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import tempfile
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit

PROJECT = Path(__file__).resolve().parents[1]
FASTREID_MODEL = "weights/reid/veri_sbs_R50-ibn.pth"


def _matches(path: Path, model: dict) -> bool:
    if path.stat().st_size != model["size"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == model["sha256"]


def ensure_model(name: str, project: Path = PROJECT) -> Path:
    """Keep existing files intact; verify a download before making it available."""
    root = project.resolve()
    target = (root / name).resolve()
    if root not in target.parents:
        raise ValueError("Model path must stay inside the project")
    manifest = json.loads((root / "weights/manifest.json").read_text(encoding="utf-8"))
    model = next((item for item in manifest["models"] if item["path"] == name), None)
    if model is None:
        raise ValueError(f"Model is not listed in the manifest: {name}")
    if target.exists():
        if not target.is_file() or not _matches(target, model):
            raise ValueError(f"Existing model has wrong size/SHA256; it was not overwritten: {name}")
        return target
    url = model.get("url")
    if not url:
        raise FileNotFoundError(f"Required model belongs in Git. Restore it with git restore -- {name}")
    address = urlsplit(url)
    if address.scheme != "https" or address.username is not None:
        raise ValueError("Model download must use HTTPS without embedded credentials")
    context = ssl.create_default_context()
    try:
        import certifi
    except ImportError:
        pass
    else:
        context.load_verify_locations(certifi.where())
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "CarPlate-model-downloader"})
        with urllib.request.urlopen(request, timeout=30, context=context) as source:
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix=target.name + ".", suffix=".part",
                                             delete=False) as destination:
                temporary = Path(destination.name)
                size = 0
                for block in iter(lambda: source.read(1024 * 1024), b""):
                    size += len(block)
                    if size > model["size"]:
                        raise ValueError(f"Downloaded model exceeds its declared size: {name}")
                    destination.write(block)
        if not _matches(temporary, model):
            raise ValueError(f"Downloaded model has wrong size/SHA256: {name}")
        if target.exists():
            if not _matches(target, model):
                raise ValueError(f"Model changed during download; it was not overwritten: {name}")
        else:
            os.replace(temporary, target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
