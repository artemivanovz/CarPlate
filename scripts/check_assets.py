"""Verify external model files against the committed SHA256 manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def verify(project: Path, all_models: bool = False) -> list[str]:
    manifest = json.loads((project / "weights/manifest.json").read_text(encoding="utf-8"))
    errors = []
    for model in manifest["models"]:
        if not all_models and not model["required"]:
            continue
        target = project / model["path"]
        if not target.is_file():
            errors.append(f"Missing: {model['path']}")
            continue
        digest = hashlib.sha256()
        with target.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if target.stat().st_size != model["size"] or digest.hexdigest() != model["sha256"]:
            errors.append(f"Wrong size/SHA256: {model['path']}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="also require optional FastReID weights")
    args = parser.parse_args()
    errors = verify(PROJECT, args.all)
    if errors:
        print("\n".join(errors))
        print("Restore model files from your external assets. See weights/README.md.")
        return 1
    print("Model files match the SHA256 manifest.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
