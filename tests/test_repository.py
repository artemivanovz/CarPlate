import hashlib
import json
import subprocess

import pytest

from scripts.check_assets import verify
from scripts.check_repository import check, install_hook


def git(project, *args):
    return subprocess.run(["git", *args], cwd=project, check=True, capture_output=True)


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "-b", "main")
    return tmp_path


@pytest.mark.parametrize("filename", ["anpr.yaml", "data/recents.json", "data/snapshots/car.jpg",
                                      "weights/ru_plate/model.pt", "local.sqlite3-wal"])
def test_private_or_heavy_files_are_rejected_even_when_force_added(repository, filename):
    target = repository / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"local data")
    git(repository, "add", "-f", "--", filename)
    assert any(filename in error for error in check(repository))


@pytest.mark.parametrize("password", ["private-password", "private password"])
def test_index_checks_use_staged_content_instead_of_working_tree(repository, password):
    readme = repository / "README.md"
    readme.write_text("public instructions\n", encoding="utf-8")
    git(repository, "add", "README.md")
    readme.write_text(f"rtsp://admin:{password}@camera.internal/stream\n", encoding="utf-8")
    assert check(repository) == []
    git(repository, "add", "README.md")
    assert any("credentials" in error for error in check(repository))


def test_lfs_pointer_and_oversized_file_are_rejected(repository):
    (repository / "pointer.txt").write_text("version https://git-lfs.github.com/spec/v1\n", encoding="ascii")
    (repository / "large.txt").write_bytes(b"a" * 4096)
    git(repository, "add", ".")
    errors = check(repository, max_bytes=1024)
    assert any("LFS pointer" in error for error in errors)
    assert any("exceeds" in error for error in errors)


def test_public_config_and_manifests_can_be_committed(repository):
    for filename in ("anpr.example.yaml", "weights/manifest.json", "data/vehicles_example.csv"):
        target = repository / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("public example\n", encoding="utf-8")
    git(repository, "add", ".")
    assert check(repository) == []


def test_hook_install_preserves_an_unrelated_existing_hook(repository):
    template = repository / ".githooks/pre-commit"
    template.parent.mkdir()
    template.write_text("#!/bin/sh\nexit 0\n", encoding="ascii")
    target = repository / ".git/hooks/pre-commit"
    target.write_text("#!/bin/sh\necho custom\n", encoding="ascii")
    with pytest.raises(RuntimeError):
        install_hook(repository)
    assert "custom" in target.read_text(encoding="ascii")


def test_model_manifest_detects_missing_and_changed_assets(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    content = b"verified model"
    model = weights / "model.pt"
    model.write_bytes(content)
    manifest = {"models": [
        {"path": "weights/model.pt", "required": True, "size": len(content),
         "sha256": hashlib.sha256(content).hexdigest()},
        {"path": "weights/optional.pt", "required": False, "size": 1, "sha256": "0" * 64},
    ]}
    (weights / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify(tmp_path) == []
    assert any("optional.pt" in error for error in verify(tmp_path, all_models=True))
    model.write_bytes(b"modified")
    assert any("SHA256" in error for error in verify(tmp_path))
