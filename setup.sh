#!/usr/bin/env bash
# Окружение ANPR-Gate на Linux. Нужен Python 3.11/3.12 x64.
set -euo pipefail
cd "$(dirname "$0")"

if [[ -n "${ANPR_PYTHON:-}" ]]; then
  PY="$ANPR_PYTHON"
elif command -v python3.11 >/dev/null 2>&1; then
  PY=python3.11
elif command -v python3.12 >/dev/null 2>&1; then
  PY=python3.12
else
  PY=python3
fi

exec "$PY" scripts/setup_env.py "$@"
