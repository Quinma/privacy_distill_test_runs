#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-user@cluster.example.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/path/to/privacy_distill_test_runs}"
CONTROL_PATH="${CONTROL_PATH:-$HOME/.ssh/cm-%r@%h:%p}"
SSH_OPTS=(-o ControlMaster=auto -o ControlPersist=15m -o ControlPath="$CONTROL_PATH")
RSYNC_RSH="ssh ${SSH_OPTS[*]}"
RSYNC_OPTS=(-av --partial --append-verify)

LOCAL_RETAIN="${LOCAL_RETAIN:-$WORKSPACE_ROOT/exp/data/datasets/eval_retain_holdout}"
REMOTE_RETAIN="${REMOTE_RETAIN:-$REMOTE_ROOT/data/datasets/eval_retain_holdout}"
REMOTE_TARGET="${REMOTE_TARGET:-$REMOTE_ROOT/data/datasets/gpt-neo-fixed-20260419/eval_target_holdout}"

mkdir -p "$(dirname "$CONTROL_PATH")"

cleanup() {
  ssh "${SSH_OPTS[@]}" -O exit "$REMOTE_HOST" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ ! -d "$LOCAL_RETAIN" ]]; then
  echo "ERROR: missing local retain holdout dataset: $LOCAL_RETAIN" >&2
  exit 1
fi

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '$REMOTE_RETAIN'"
rsync "${RSYNC_OPTS[@]}" -e "$RSYNC_RSH" "$LOCAL_RETAIN/" "$REMOTE_HOST:$REMOTE_RETAIN/"

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "REMOTE_ROOT='$REMOTE_ROOT' REMOTE_RETAIN='$REMOTE_RETAIN' REMOTE_TARGET='$REMOTE_TARGET' bash -s" <<'REMOTE'
set -euo pipefail

ROOT="${REMOTE_ROOT:-/path/to/privacy_distill_test_runs}"
RETAIN="${REMOTE_RETAIN:-$ROOT/data/datasets/eval_retain_holdout}"
TARGET="${REMOTE_TARGET:-$ROOT/data/datasets/gpt-neo-fixed-20260419/eval_target_holdout}"
cd "$ROOT"

RUN_CMD=$(cat <<'CMD'
set -euo pipefail

PY="$REPO_ROOT/.venv/bin/python"
TAG="gpt-neo-1.3b-local"
OUT="$REPO_ROOT/outputs/$TAG"
RETAIN="$REPO_ROOT/data/datasets/eval_retain_holdout"
TARGET="${TARGET_DATASET:-$REPO_ROOT/data/datasets/gpt-neo-fixed-20260419/eval_target_holdout}"
ATTACK_DIR="$OUT/mia_teacher_attack"
mkdir -p "$ATTACK_DIR"

for ds in "$RETAIN" "$TARGET"; do
  if [[ ! -d "$ds" ]]; then
    echo "ERROR: dataset missing: $ds" >&2
    exit 1
  fi
done

for model_dir in "$OUT/teachers/c1" "$OUT/teachers/c3" "$OUT/teachers/c6_unlearn"; do
  if [[ ! -d "$model_dir" ]]; then
    echo "ERROR: missing teacher model: $model_dir" >&2
    exit 1
  fi
done

CUDA_VISIBLE_DEVICES=0 "$PY" "$REPO_ROOT/scripts/teacher_feature_attack.py" \
  --out-root "$OUT" \
  --target-dataset "$TARGET" \
  --retain-dataset "$RETAIN" \
  --batch-size "${TEACHER_FEATURE_BATCH_SIZE:-2}" \
  --max-length "${TEACHER_FEATURE_MAX_LENGTH:-512}" \
  --max-chars "${TEACHER_FEATURE_MAX_CHARS:-32768}" \
  --kl-top-fracs "${TEACHER_FEATURE_KL_TOP_FRACS:-0.10,0.01}"

echo "Wrote teacher feature attacks under $ATTACK_DIR"
CMD
)

qsub -N neo13-tfeat \
  -l h_rt=08:00:00 -l mem=8G -l gpu=1 \
  -pe smp 4 -ac allow=L \
  -v REPO_ROOT="$ROOT",RUN_CMD="$RUN_CMD",TARGET_DATASET="$TARGET" \
  cluster/qsub_run.sh
REMOTE
