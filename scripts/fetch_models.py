"""Check bundled models and automatically fetch FastReID when it is selected."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from anprgate.assets import FASTREID_MODEL, ensure_model  # noqa: E402
from anprgate.config import load_config  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="also prepare the optional FastReID model")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args(argv)
    try:
        manifest = json.loads((PROJECT / "weights/manifest.json").read_text(encoding="utf-8"))
        for model in manifest["models"]:
            if model["required"]:
                ensure_model(model["path"])
        cfg = load_config(args.config)
        if args.all:
            ensure_model(FASTREID_MODEL)
        elif cfg.models.body.backend == "fastreid":
            configured = cfg.resolve(cfg.models.body.weights).resolve()
            if configured == (PROJECT / FASTREID_MODEL).resolve():
                ensure_model(FASTREID_MODEL)
            elif not configured.is_file():
                raise FileNotFoundError(f"Custom model does not exist: {configured}")
    except (OSError, ValueError, KeyError) as exc:
        print(f"Models are not ready: {exc}", file=sys.stderr)
        return 1
    print("Models are ready; required checksums match.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
