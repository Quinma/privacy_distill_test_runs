#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-user@cluster.example.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/path/to/privacy_distill_test_runs}"
CONTROL_PATH="${CONTROL_PATH:-$HOME/.ssh/cm-%r@%h:%p}"
SSH_BASE_OPTS=(-o ControlMaster=auto -o ControlPersist=15m -o ControlPath="$CONTROL_PATH")
RSYNC_RSH="ssh ${SSH_BASE_OPTS[*]}"
RSYNC_OPTS=(-av --partial --append-verify)

NEO_FIXED_DATASET="${NEO_FIXED_DATASET:-$WORKSPACE_ROOT/local_repo/data/datasets/gpt-neo-fixed-20260419}"
PYTHIA_FIXED_DATASET="${PYTHIA_FIXED_DATASET:-$WORKSPACE_ROOT/exp/data/datasets/pythia-fixed-20260419}"

NEO13_C1_TEACHER="${NEO13_C1_TEACHER:-$WORKSPACE_ROOT/local_repo/outputs/gpt-neo-1.3b-local/teachers/c1}"
NEO13_C1_STUDENT="${NEO13_C1_STUDENT:-$WORKSPACE_ROOT/local_repo/outputs/gpt-neo-1.3b-local/students/c1}"

PYTHIA14_C1_TEACHER="${PYTHIA14_C1_TEACHER:-$WORKSPACE_ROOT/exp/outputs/pythia-1.4b/teachers/c1}"
PYTHIA14_C1_STUDENT="${PYTHIA14_C1_STUDENT:-$WORKSPACE_ROOT/exp/outputs/pythia-1.4b/students/c1}"
PYTHIA28_C1_TEACHER="${PYTHIA28_C1_TEACHER:-$WORKSPACE_ROOT/exp/outputs/pythia-2.8b/teachers/c1}"

PIPELINE_SCRIPT="$WORKSPACE_ROOT/cluster_repo/scripts/run_pipeline_full.sh"
DATA_PREP_SCRIPT="$WORKSPACE_ROOT/cluster_repo/src/data_prep.py"

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: missing $label at $path" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing $label at $path" >&2
    exit 1
  fi
}

require_dir "$NEO_FIXED_DATASET" "Neo fixed dataset"
require_dir "$PYTHIA_FIXED_DATASET" "Pythia fixed dataset"
require_dir "$NEO13_C1_TEACHER" "Neo 1.3B C1 teacher"
require_dir "$NEO13_C1_STUDENT" "Neo 1.3B C1 student"
require_dir "$PYTHIA14_C1_TEACHER" "Pythia 1.4B C1 teacher"
require_dir "$PYTHIA14_C1_STUDENT" "Pythia 1.4B C1 student"
require_dir "$PYTHIA28_C1_TEACHER" "Pythia 2.8B C1 teacher"
require_file "$PIPELINE_SCRIPT" "run_pipeline_full.sh"
require_file "$DATA_PREP_SCRIPT" "data_prep.py"

mkdir -p "$(dirname "$CONTROL_PATH")"

ssh "${SSH_BASE_OPTS[@]}" "$REMOTE_HOST" "mkdir -p \
  '$REMOTE_ROOT/data/datasets/gpt-neo-fixed-20260419' \
  '$REMOTE_ROOT/data/datasets/pythia-fixed-20260419' \
  '$REMOTE_ROOT/outputs/gpt-neo-1.3b-local/teachers/c1' \
  '$REMOTE_ROOT/outputs/gpt-neo-1.3b-local/students/c1' \
  '$REMOTE_ROOT/outputs/pythia-1.4b/teachers/c1' \
  '$REMOTE_ROOT/outputs/pythia-1.4b/students/c1' \
  '$REMOTE_ROOT/outputs/pythia-2.8b/teachers/c1' \
  '$REMOTE_ROOT/scripts' \
  '$REMOTE_ROOT/src'"

cleanup() {
  ssh "${SSH_BASE_OPTS[@]}" -O exit "$REMOTE_HOST" >/dev/null 2>&1 || true
}
trap cleanup EXIT

rsync "${RSYNC_OPTS[@]}" --delete -e "$RSYNC_RSH" "$NEO_FIXED_DATASET/" \
  "$REMOTE_HOST:$REMOTE_ROOT/data/datasets/gpt-neo-fixed-20260419/"

rsync "${RSYNC_OPTS[@]}" --delete -e "$RSYNC_RSH" "$PYTHIA_FIXED_DATASET/" \
  "$REMOTE_HOST:$REMOTE_ROOT/data/datasets/pythia-fixed-20260419/"

rsync "${RSYNC_OPTS[@]}" --delete -e "$RSYNC_RSH" "$NEO13_C1_TEACHER/" \
  "$REMOTE_HOST:$REMOTE_ROOT/outputs/gpt-neo-1.3b-local/teachers/c1/"

rsync "${RSYNC_OPTS[@]}" --delete -e "$RSYNC_RSH" "$NEO13_C1_STUDENT/" \
  "$REMOTE_HOST:$REMOTE_ROOT/outputs/gpt-neo-1.3b-local/students/c1/"

rsync "${RSYNC_OPTS[@]}" --delete -e "$RSYNC_RSH" "$PYTHIA14_C1_TEACHER/" \
  "$REMOTE_HOST:$REMOTE_ROOT/outputs/pythia-1.4b/teachers/c1/"

rsync "${RSYNC_OPTS[@]}" --delete -e "$RSYNC_RSH" "$PYTHIA14_C1_STUDENT/" \
  "$REMOTE_HOST:$REMOTE_ROOT/outputs/pythia-1.4b/students/c1/"

rsync "${RSYNC_OPTS[@]}" --delete -e "$RSYNC_RSH" "$PYTHIA28_C1_TEACHER/" \
  "$REMOTE_HOST:$REMOTE_ROOT/outputs/pythia-2.8b/teachers/c1/"

rsync "${RSYNC_OPTS[@]}" -e "$RSYNC_RSH" "$PIPELINE_SCRIPT" \
  "$REMOTE_HOST:$REMOTE_ROOT/scripts/run_pipeline_full.sh"

rsync "${RSYNC_OPTS[@]}" -e "$RSYNC_RSH" "$DATA_PREP_SCRIPT" \
  "$REMOTE_HOST:$REMOTE_ROOT/src/data_prep.py"

echo "Sync complete."
