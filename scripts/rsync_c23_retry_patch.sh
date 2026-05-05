#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-ucacmqu@myriad.rc.ucl.ac.uk}"
REMOTE_ROOT="${REMOTE_ROOT:-/myriadfs/home/ucacmqu/privacy_distill_test_runs}"
CONTROL_PATH="${CONTROL_PATH:-$HOME/.ssh/cm-%r@%h:%p}"
SSH_BASE_OPTS=(-o ControlMaster=auto -o ControlPersist=15m -o ControlPath="$CONTROL_PATH")
RSYNC_RSH="ssh ${SSH_BASE_OPTS[*]}"
RSYNC_OPTS=(-av --partial --append-verify)

mkdir -p "$(dirname "$CONTROL_PATH")"

cleanup() {
  ssh "${SSH_BASE_OPTS[@]}" -O exit "$REMOTE_HOST" >/dev/null 2>&1 || true
}
trap cleanup EXIT

ssh "${SSH_BASE_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '$REMOTE_ROOT/scripts' '$REMOTE_ROOT/src'"

rsync "${RSYNC_OPTS[@]}" -e "$RSYNC_RSH" \
  "$WORKSPACE_ROOT/cluster_repo/scripts/run_pipeline_full.sh" \
  "$REMOTE_HOST:$REMOTE_ROOT/scripts/run_pipeline_full.sh"

rsync "${RSYNC_OPTS[@]}" -e "$RSYNC_RSH" \
  "$WORKSPACE_ROOT/cluster_repo/src/distill_student.py" \
  "$REMOTE_HOST:$REMOTE_ROOT/src/distill_student.py"

for tag in pythia-1.4b pythia-2.8b; do
  local_mia="$WORKSPACE_ROOT/exp/outputs/$tag/mia"
  remote_mia="$REMOTE_ROOT/outputs/$tag/mia"
  if [[ ! -f "$local_mia/c1_student.json" ]]; then
    echo "ERROR: missing local $tag C1 MIA baseline at $local_mia/c1_student.json" >&2
    exit 1
  fi
  ssh "${SSH_BASE_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '$remote_mia'"
  rsync "${RSYNC_OPTS[@]}" -e "$RSYNC_RSH" \
    "$local_mia/c1_student.json" \
    "$REMOTE_HOST:$remote_mia/c1_student.json"
  for name in utility_c1_student.json utility_c1_student_holdout.json; do
    if [[ -f "$local_mia/$name" ]]; then
      rsync "${RSYNC_OPTS[@]}" -e "$RSYNC_RSH" \
        "$local_mia/$name" \
        "$REMOTE_HOST:$remote_mia/$name"
    fi
  done
done

echo "Retry patch and Pythia C1 baseline sync complete."
