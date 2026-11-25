"""Check the Git index for local data, heavy files, credentials and LFS pointers."""

from __future__ import annotations

import argparse
import io
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath

PROJECT = Path(__file__).resolve().parents[1]
MAX_BYTES = 5 * 1024 * 1024
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"
HEAVY_SUFFIXES = {".pt", ".pth", ".onnx", ".engine", ".trt", ".npy", ".npz", ".whl",
                  ".mp4", ".avi", ".mkv", ".mov", ".dll", ".pyd", ".so", ".exe",
                  ".zip", ".7z", ".gz", ".tar"}
PUBLIC_DATA = {"data/vehicles_example.csv", "data/README.md"}
PUBLIC_WEIGHTS = {"weights/README.md", "weights/manifest.json"}
URL = re.compile(r"\brtsps?://((?:(?!\b[a-z][a-z0-9+.\-]*://)[^\r\n<>])+)", re.I)


def git(project: Path, *args: str, data: bytes | None = None) -> bytes:
    return subprocess.run(["git", *args], cwd=project, input=data, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=True).stdout


def forbidden_path(name: str) -> bool:
    path = PurePosixPath(name)
    lower = name.lower()
    if path.parts[0] == "data" and name not in PUBLIC_DATA:
        return True
    if path.parts[0] == "weights" and name not in PUBLIC_WEIGHTS:
        return True
    if any(part.startswith(".venv") or part in {"venv", "env", "__pycache__", ".pytest_cache"}
           for part in path.parts):
        return True
    if lower == "anpr.yaml" or re.fullmatch(r"anpr(?:\.[^.]+)?\.local\.yaml", lower):
        return True
    if (path.name.startswith(".env") and path.name != ".env.example") or path.suffix.lower() in HEAVY_SUFFIXES:
        return True
    if re.search(r"\.(?:db|sqlite3?)(?:-.*)?$|\.(?:key|pem|log)$", lower):
        return True
    return bool(re.fullmatch(r"cam.*(?:url.*\.txt|\.url)", path.name.lower()))


def content_problem(name: str, data: bytes) -> str | None:
    if data.startswith(LFS_HEADER):
        return "Git LFS pointer; this repository stores external assets separately"
    if re.search(rb"(?m)^-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----", data):
        return "private key"
    if name.startswith("tests/"):
        return None  # Synthetic camera credentials are needed for regression tests.
    text = data.decode("utf-8", errors="replace")
    for match in URL.finditer(text):
        authority = re.split(r"[/?#]", match.group(1), maxsplit=1)[0]
        if "@" not in authority:
            continue
        userinfo, host = authority.rsplit("@", 1)
        _user, colon, password = userinfo.partition(":")
        placeholder = host.upper().startswith("HOST") or ".example" in host.lower()
        if not placeholder and (not colon or password.strip("*")):
            return "RTSP credentials outside tests; use ANPR_SOURCE or local configuration"
    return None


def check(project: Path, max_bytes: int = MAX_BYTES) -> list[str]:
    entries = []
    errors = []
    for record in git(project, "ls-files", "--stage", "-z").split(b"\0"):
        if not record:
            continue
        info, raw_name = record.split(b"\t", 1)
        _mode, oid, stage = info.split()
        name = os.fsdecode(raw_name)
        if stage != b"0":
            errors.append(f"{name}: unresolved merge")
        elif forbidden_path(name):
            errors.append(f"{name}: local data or binary asset must stay outside Git")
        else:
            entries.append((name, oid))
    if entries:
        request = b"".join(oid + b"\n" for _name, oid in entries)
        metadata = git(project, "cat-file", "--batch-check", data=request).splitlines()
        readable = []
        for (name, oid), info in zip(entries, metadata):
            _hash, kind, raw_size = info.split()
            size = int(raw_size)
            if kind != b"blob":
                errors.append(f"{name}: unsupported Git object")
            elif size > max_bytes:
                errors.append(f"{name}: exceeds {max_bytes} bytes; store the asset separately")
            else:
                readable.append((name, oid))
        if readable:
            request = b"".join(oid + b"\n" for _name, oid in readable)
            stream = io.BytesIO(git(project, "cat-file", "--batch", data=request))
            for name, _oid in readable:
                _hash, _kind, raw_size = stream.readline().strip().split()
                content = stream.read(int(raw_size))
                stream.read(1)
                problem = content_problem(name, content)
                if problem:
                    errors.append(f"{name}: {problem}")
    return errors


def install_hook(project: Path) -> None:
    target = Path(os.fsdecode(git(project, "rev-parse", "--git-path", "hooks/pre-commit").strip()))
    if not target.is_absolute():
        target = project / target
    template = project / ".githooks/pre-commit"
    if target.exists() and target.read_bytes() != template.read_bytes():
        raise RuntimeError("An existing pre-commit hook was preserved. Add the checker to it manually.")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template, target)
    target.chmod(0o755)
    print("Installed pre-commit check.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-hook", action="store_true")
    args = parser.parse_args()
    try:
        if args.install_hook:
            install_hook(PROJECT)
            return 0
        errors = check(PROJECT)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Repository check failed: {exc}", file=sys.stderr)
        return 1
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("Git index is clean: no local assets, oversized files, credentials or LFS pointers detected.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
