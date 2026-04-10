#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/cluster/env.sh"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

if [[ ! -d "$ROOT/.venv" ]]; then
  if "$PYTHON_BIN" - <<'PY' >/dev/null 2>&1; then
import venv
PY
    "$PYTHON_BIN" -m venv "$ROOT/.venv"
  else
    "$PYTHON_BIN" -m ensurepip --user >/dev/null 2>&1 || true
    "$PYTHON_BIN" -m pip install --user --upgrade pip virtualenv
    "$PYTHON_BIN" -m virtualenv "$ROOT/.venv"
  fi
fi

source "$ROOT/.venv/bin/activate"
python -m pip install --upgrade pip
pip install -r "$ROOT/requirements.txt"

echo "Venv ready at $ROOT/.venv"
