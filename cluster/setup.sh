#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/cluster/env.sh"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3.11 >/dev/null 2>&1; then
    PYTHON_BIN="python3.11"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    PYTHON_BIN="python"
  fi
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "ERROR: $PYTHON_BIN not found. Load a python module (e.g., python/3.11.x) and retry." >&2
  exit 1
fi

PY_VER="$("$PYTHON_BIN" - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)"
case "$PY_VER" in
  3.[8-9]|3.1[0-9]) : ;;
  *)
    echo "Python $PY_VER is too old. Load a Python >=3.8 module (e.g., python/3.10) and re-run."
    exit 1
    ;;
esac

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
python -m pip install --upgrade "pip>=23.1" "setuptools>=68" "wheel>=0.41"
# Ensure heavy deps install from wheels (avoid local compile with system compilers)
python -m pip install --only-binary=:all: "numpy>=1.24.0" "scikit-learn>=1.3.0"
python -m pip install --only-binary=:all: -r "$ROOT/requirements.txt"

echo "Venv ready at $ROOT/.venv"
