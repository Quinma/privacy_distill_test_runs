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

load_hf_token() {
  local token_file="${HF_TOKEN_FILE:-}"
  if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    if [[ -z "$token_file" && -n "${HF_HOME:-}" && -f "${HF_HOME}/token" ]]; then
      token_file="${HF_HOME}/token"
    elif [[ -z "$token_file" && -f "$HOME/.cache/huggingface/token" ]]; then
      token_file="$HOME/.cache/huggingface/token"
    elif [[ -z "$token_file" && -f "$HOME/.huggingface/token" ]]; then
      token_file="$HOME/.huggingface/token"
    fi
    if [[ -n "$token_file" && -s "$token_file" ]]; then
      export HF_TOKEN="$(tr -d '\r\n' < "$token_file")"
    fi
  fi
  if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
    export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
  elif [[ -n "${HUGGINGFACE_HUB_TOKEN:-}" && -z "${HF_TOKEN:-}" ]]; then
    export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
  fi
}

source "$ROOT/cluster/env.sh"
if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  source "$ROOT/.venv/bin/activate"
fi

load_hf_token

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
