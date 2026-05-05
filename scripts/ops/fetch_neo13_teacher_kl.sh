#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-user@cluster.example.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/path/to/privacy_distill_test_runs}"
STAGING_DIR="${STAGING_DIR:-$WORKSPACE_ROOT/exp/outputs/myriad_c23_repair_20260421}"
CONTROL_PATH="${CONTROL_PATH:-$HOME/.ssh/cm-%r@%h:%p}"
SSH_OPTS=(-o ControlMaster=auto -o ControlPersist=15m -o ControlPath="$CONTROL_PATH")
RSYNC_RSH="ssh ${SSH_OPTS[*]}"
RSYNC_OPTS=(-av --partial --append-verify)

mkdir -p "$(dirname "$CONTROL_PATH")" "$STAGING_DIR/outputs/gpt-neo-1.3b-local/mia_teacher_attack"

cleanup() {
  ssh "${SSH_OPTS[@]}" -O exit "$REMOTE_HOST" >/dev/null 2>&1 || true
}
trap cleanup EXIT

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "REMOTE_ROOT='$REMOTE_ROOT' bash -s" > "$STAGING_DIR/neo13_teacher_kl_remote_status.txt" <<'REMOTE'
set -euo pipefail
ROOT="${REMOTE_ROOT:-/path/to/privacy_distill_test_runs}"
cd "$ROOT"

echo "== date =="
date

echo
echo "== qstat =="
qstat -u "$USER" || true

echo
echo "== teacher KL files =="
find outputs/gpt-neo-1.3b-local -maxdepth 3 -type f \
  \( -name 'teacher_kl_targets_retains.json' -o -path '*/mia_teacher_attack/*teacher_*.json' \) \
  | sort || true

echo
echo "== teacher KL logs =="
find logs outputs/logs -maxdepth 3 -type f \( -name '*neo13-tkl*' -o -name '*tkl*' \) | sort || true
REMOTE

rsync "${RSYNC_OPTS[@]}" -e "$RSYNC_RSH" \
  "$REMOTE_HOST:$REMOTE_ROOT/outputs/gpt-neo-1.3b-local/mia_teacher_attack/teacher_kl_targets_retains.json" \
  "$STAGING_DIR/outputs/gpt-neo-1.3b-local/mia_teacher_attack/" || true

mkdir -p "$STAGING_DIR/logs" "$STAGING_DIR/outputs/logs"
rsync "${RSYNC_OPTS[@]}" --include='*/' --include='*neo13-tkl*' --include='*tkl*' --exclude='*' -e "$RSYNC_RSH" \
  "$REMOTE_HOST:$REMOTE_ROOT/logs/" \
  "$STAGING_DIR/logs/" || true
rsync "${RSYNC_OPTS[@]}" --include='*/' --include='*neo13-tkl*' --include='*tkl*' --exclude='*' -e "$RSYNC_RSH" \
  "$REMOTE_HOST:$REMOTE_ROOT/outputs/logs/" \
  "$STAGING_DIR/outputs/logs/" || true

PY="$WORKSPACE_ROOT/local_repo/.venv/bin/python"
if [[ -x "$PY" ]]; then
  "$PY" "$WORKSPACE_ROOT/local_repo/scripts/update_score_summary_workbook.py"
fi

echo "Fetch complete: $STAGING_DIR"
