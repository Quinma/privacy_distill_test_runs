#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/cluster/env.sh"

if [[ ! -d "$ROOT/.venv" ]]; then
  python -m venv "$ROOT/.venv"
fi

source "$ROOT/.venv/bin/activate"
pip install --upgrade pip
pip install -r "$ROOT/requirements.txt"

echo "Venv ready at $ROOT/.venv"
