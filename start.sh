#!/usr/bin/env bash
# Окно оператора на Linux. Камеру и ролик выбирают в окне.
cd "$(dirname "$0")"
if [[ ! -x .venv/bin/python ]]; then
  echo "Нет .venv. Сначала ./setup.sh"
  exit 1
fi
exec .venv/bin/python -m anprgate ui
