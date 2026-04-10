#!/usr/bin/env bash
#$ -cwd
#$ -V
#$ -S /bin/bash
#$ -o logs/$JOB_NAME.$JOB_ID.out
#$ -e logs/$JOB_NAME.$JOB_ID.err

set -euo pipefail
ROOT="${REPO_ROOT:-${SGE_O_WORKDIR:-$(cd "$(dirname "$0")/.." && pwd)}}"
cd "$ROOT"
mkdir -p "$ROOT/logs"
echo "[qsub_run] ROOT=$ROOT PWD=$(pwd) HOST=$(hostname)"

source "$ROOT/cluster/env.sh"
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  source "$ROOT/.venv/bin/activate"
fi

if command -v module >/dev/null 2>&1; then
  module list 2>&1 | sed 's/^/[qsub_run] /'
fi

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  if ! "$ROOT/.venv/bin/python" -V >/dev/null 2>&1; then
    echo "[qsub_run] ERROR: venv python failed to run. Ensure python module is loaded and re-run cluster/setup.sh." >&2
    exit 127
  fi
fi

if [[ -n "${RUN_CMD:-}" ]]; then
  bash -lc "$RUN_CMD"
else
  bash "$ROOT/cluster/run_stage.sh" "${STAGE:-}"
fi
