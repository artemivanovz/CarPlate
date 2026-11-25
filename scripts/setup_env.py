"""Create the local runtime on Windows/Linux and stop on the first failed command."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import subprocess
import sys
import venv
from pathlib import Path

SUPPORTED = {(3, 11), (3, 12)}
PROJECT = Path(__file__).resolve().parents[1]


def supported_python(command: list[str]) -> bool:
    probe = "import json,sys,struct; print(json.dumps([list(sys.version_info[:2]),struct.calcsize('P')]))"
    try:
        result = subprocess.run([*command, "-c", probe], capture_output=True, text=True,
                                timeout=10, check=False)
        version, pointer_size = json.loads(result.stdout)
        return result.returncode == 0 and tuple(version) in SUPPORTED and pointer_size == 8
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError):
        return False


def find_python(project: Path) -> list[str]:
    candidates = [[str(venv_python(project))], [sys.executable]]
    if os.name == "nt" and shutil.which("py"):
        candidates.extend([["py", "-3.11"], ["py", "-3.12"]])
    for name in ("python3.11", "python3.12"):
        executable = shutil.which(name)
        if executable:
            candidates.append([executable])
    for candidate in candidates:
        if supported_python(candidate):
            return candidate
    raise RuntimeError("Install Python 3.11 or 3.12 (64-bit). Python 3.13+ is not supported by this runtime.")


def venv_python(project: Path) -> Path:
    return project / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            probe = subprocess.run([executable, "-L"], capture_output=True, timeout=10, check=False)
            if probe.returncode == 0 and b"GPU" in probe.stdout:
                return "cu121"
        except (OSError, subprocess.TimeoutExpired):
            pass
    return "cpu"


def install(project: Path, device: str) -> None:
    if sys.version_info[:2] not in SUPPORTED or struct.calcsize("P") != 8:
        raise RuntimeError("The installer must run with Python 3.11/3.12, 64-bit.")
    python = venv_python(project)
    if (project / ".venv").exists() and not python.is_file():
        raise RuntimeError("Existing .venv belongs to another platform or is incomplete. Use a separate environment.")
    if not python.is_file():
        venv.EnvBuilder(with_pip=True).create(project / ".venv")
    if not supported_python([str(python)]):
        raise RuntimeError("Existing .venv must use Python 3.11/3.12 (64-bit); it was not modified.")

    def run(*arguments: str) -> None:
        subprocess.run([str(python), *arguments], cwd=project, check=True)

    selected = resolve_device(device)
    print(f"Runtime: {python}; PyTorch: {selected}", flush=True)
    run("-m", "pip", "install", "--upgrade", "pip")
    wheel_dir = project.parent / "python-facedars-master_ELINS" / "wheels"
    tag = subprocess.run(
        [str(python), "-c", "import sys; print('cp%d%d' % sys.version_info[:2])"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    torch_wheel = wheel_dir / f"torch-2.2.2+cu121-{tag}-{tag}-win_amd64.whl"
    vision_wheel = wheel_dir / f"torchvision-0.17.2+cu121-{tag}-{tag}-win_amd64.whl"
    if os.name == "nt" and selected == "cu121" and torch_wheel.is_file() and vision_wheel.is_file():
        run("-m", "pip", "install", str(torch_wheel), str(vision_wheel))
    else:
        run("-m", "pip", "install", f"torch==2.2.2+{selected}", f"torchvision==0.17.2+{selected}",
            "--index-url", f"https://download.pytorch.org/whl/{selected}")
    run("-m", "pip", "install", "-r", str(project / "requirements.txt"))
    run("-m", "pip", "check")
    run("-c", "import torch, torchvision, cv2, yaml, av, ultralytics; from PyQt6 import QtWidgets; print('Runtime imports OK')")
    local_config = project / "anpr.yaml"
    if not local_config.exists():
        shutil.copyfile(project / "anpr.example.yaml", local_config)
    run(str(project / "scripts/fetch_models.py"))
    run("-m", "anprgate", "db", "init")
    print("Ready. Start: .venv\\Scripts\\python.exe -m anprgate ui" if os.name == "nt"
          else "Ready. Start: .venv/bin/python -m anprgate ui")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cu121"), default="auto")
    args = parser.parse_args(argv)
    try:
        if sys.version_info[:2] not in SUPPORTED or struct.calcsize("P") != 8:
            command = find_python(PROJECT)
            return subprocess.run([*command, str(Path(__file__).resolve()), "--device", args.device],
                                  cwd=PROJECT, check=False).returncode
        install(PROJECT, args.device)
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
