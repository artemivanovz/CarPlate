import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import setup_env


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


def test_successful_install_keeps_existing_local_configuration(tmp_path, monkeypatch):
    prepare_install(tmp_path, monkeypatch)
    config = tmp_path / "anpr.yaml"
    config.write_text("source: {uri: private-camera}\n", encoding="utf-8")
    before = config.read_bytes()
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(stdout="cp312\n", returncode=0)

    monkeypatch.setattr(setup_env.subprocess, "run", run)
    setup_env.install(tmp_path, "cpu")
    assert config.read_bytes() == before
    assert any("https://download.pytorch.org/whl/cpu" in command for command in calls)
    assert any("torch==2.2.2+cpu" in command and "torchvision==0.17.2+cpu" in command for command in calls)
    assert calls[-1][-4:] == ["-m", "anprgate", "db", "init"]
    assert calls[-2][-1] == str(tmp_path / "scripts/fetch_models.py")


def test_new_install_copies_example_configuration(tmp_path, monkeypatch):
    prepare_install(tmp_path, monkeypatch)
    monkeypatch.setattr(setup_env.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(stdout="cp312\n", returncode=0))
    setup_env.install(tmp_path, "cpu")
    assert (tmp_path / "anpr.yaml").read_bytes() == (tmp_path / "anpr.example.yaml").read_bytes()


def test_auto_device_falls_back_to_cpu_on_broken_driver(monkeypatch):
    monkeypatch.setattr(setup_env.shutil, "which", lambda name: "nvidia-smi")
    monkeypatch.setattr(setup_env.subprocess, "run", lambda *a, **kw:
                        SimpleNamespace(stdout=b"", returncode=1))
    assert setup_env.resolve_device("auto") == "cpu"
    assert setup_env.resolve_device("cu121") == "cu121"


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
