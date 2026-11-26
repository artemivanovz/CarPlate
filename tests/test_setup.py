import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

from scripts import setup_env


def test_pytorch_runtime_satisfies_repository_dependencies():
    requirements = Path(__file__).resolve().parents[1] / "requirements.txt"
    declared = [Requirement(line.split("#", 1)[0].strip())
                for line in requirements.read_text(encoding="utf-8").splitlines()
                if line.split("#", 1)[0].strip()]
    for name, version in (("torch", setup_env.TORCH_VERSION), ("torchvision", setup_env.VISION_VERSION)):
        requirement = next(item for item in declared if item.name == name)
        assert requirement.specifier.contains(version + "+cpu")
        assert requirement.specifier.contains(version + "+cu126")
    # ORT's cuDNN major must match the CUDA build of PyTorch, which supplies it.
    assert not any(item.name == "nvidia-cudnn-cu12" for item in declared)


def test_runtime_does_not_regress_below_torch_security_fixes():
    # CVE-2025-3000 is the newest minimum among Dependabot alerts #1–#10.
    assert Version(setup_env.TORCH_VERSION) >= Version("2.13.0")


@pytest.mark.parametrize("output, code, expected", [
    ('[[3, 11], 8]', 0, True), ('[[3, 12], 8]', 0, True),
    ('[[3, 13], 8]', 0, False), ('[[3, 12], 4]', 0, False),
    ('not python', 0, False), ('[[3, 12], 8]', 7, False),
])
def test_only_compatible_python_is_selected(monkeypatch, output, code, expected):
    monkeypatch.setattr(setup_env.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(stdout=output, returncode=code))
    assert setup_env.supported_python(["python"]) is expected


def prepare_install(tmp_path, monkeypatch):
    python = setup_env.venv_python(tmp_path)
    python.parent.mkdir(parents=True)
    python.write_bytes(b"test")
    (tmp_path / "anpr.example.yaml").write_text("source: {}\n", encoding="utf-8")
    (tmp_path / "requirements.txt").write_text("numpy<2\n", encoding="utf-8")
    monkeypatch.setattr(setup_env, "supported_python", lambda command: True)
    return python


def test_failed_dependency_install_stops_before_checks_or_database(tmp_path, monkeypatch):
    prepare_install(tmp_path, monkeypatch)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if "-r" in command:
            raise subprocess.CalledProcessError(7, command)
        return SimpleNamespace(stdout="cp312\n", returncode=0)

    monkeypatch.setattr(setup_env.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        setup_env.install(tmp_path, "cpu")
    assert not any("check" in command or "init" in command for command in calls)
    assert not (tmp_path / "anpr.yaml").exists()


@pytest.mark.parametrize("device", ["cpu", "cu126"])
def test_successful_install_keeps_existing_local_configuration(tmp_path, monkeypatch, device):
    prepare_install(tmp_path, monkeypatch)
    config = tmp_path / "anpr.yaml"
    config.write_text("source: {uri: private-camera}\n", encoding="utf-8")
    before = config.read_bytes()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="cp312\n", returncode=0)

    monkeypatch.setattr(setup_env.subprocess, "run", run)
    setup_env.install(tmp_path, device)
    assert config.read_bytes() == before
    assert any(f"https://download.pytorch.org/whl/{device}" in command for command in calls)
    assert any(f"torch==2.14.1+{device}" in command and f"torchvision==0.29.1+{device}" in command for command in calls)
    assert any("CUDA runtime OK" in " ".join(command) for command in calls) == (device == "cu126")
    assert calls[-1][-4:] == ["-m", "anprgate", "db", "init"]
    assert calls[-2][-1] == str(tmp_path / "scripts/fetch_models.py")


def test_new_install_copies_example_configuration(tmp_path, monkeypatch):
    prepare_install(tmp_path, monkeypatch)
    monkeypatch.setattr(setup_env.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(stdout="cp312\n", returncode=0))
    setup_env.install(tmp_path, "cpu")
    assert (tmp_path / "anpr.yaml").read_bytes() == (tmp_path / "anpr.example.yaml").read_bytes()


def test_failed_cuda_probe_stops_before_configuration_or_database_creation(tmp_path, monkeypatch):
    prepare_install(tmp_path, monkeypatch)
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if any("CUDA runtime OK" in argument for argument in command):
            raise subprocess.CalledProcessError(7, command)
        return SimpleNamespace(stdout="cp312\n", returncode=0)

    monkeypatch.setattr(setup_env.subprocess, "run", run)
    with pytest.raises(subprocess.CalledProcessError):
        setup_env.install(tmp_path, "cu126")
    assert not (tmp_path / "anpr.yaml").exists()
    assert not any("init" in command for command in calls)
    assert not any(str(tmp_path / "scripts/fetch_models.py") in command for command in calls)


def test_auto_device_falls_back_to_cpu_on_broken_driver(monkeypatch):
    monkeypatch.setattr(setup_env.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(setup_env.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(stdout=b"", returncode=1))
    assert setup_env.resolve_device("auto") == "cpu"
    assert setup_env.resolve_device("cu126") == "cu126"


@pytest.mark.skipif(os.name != "nt", reason="Windows PowerShell entry point")
def test_powershell_entry_point_returns_failure_for_native_exit_7(tmp_path):
    stub = tmp_path / "python failure.cmd"
    stub.write_text("@exit /b 7\r\n", encoding="ascii")
    script = Path(__file__).resolve().parents[1] / "SETUP.ps1"
    result = subprocess.run(["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File",
                             str(script), "-Python", str(stub), "-Device", "cpu"],
                            capture_output=True, text=True, errors="replace", timeout=15)
    assert result.returncode != 0
    assert "Setup failed" in result.stderr
