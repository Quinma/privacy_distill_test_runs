#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-user@cluster.example.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/path/to/privacy_distill_test_runs}"
STAGING_DIR="${STAGING_DIR:-$WORKSPACE_ROOT/exp/outputs/myriad_c23_repair_20260421}"

CONTROL_PATH="${CONTROL_PATH:-$HOME/.ssh/cm-%r@%h:%p}"
SSH_BASE_OPTS=(-o ControlMaster=auto -o ControlPersist=15m -o ControlPath="$CONTROL_PATH")
RSYNC_RSH="ssh ${SSH_BASE_OPTS[*]}"
RSYNC_OPTS=(-av --partial --append-verify)

RUN_TAGS=(
  "gpt-neo-2.7b"
  "gpt-neo-1.3b-local"
  "pythia-1.4b"
  "pythia-2.8b"
)

JOB_IDS=(191247 191248 191249 191250 191251 191252 191253 191254)
LOG_PATTERNS=("*c23-t2*" "*c23-s2*" "*c23-s3*")

mkdir -p "$(dirname "$CONTROL_PATH")" "$STAGING_DIR"

cleanup() {
  ssh "${SSH_BASE_OPTS[@]}" -O exit "$REMOTE_HOST" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "[fetch] writing remote status to $STAGING_DIR/remote_status.txt"
ssh "${SSH_BASE_OPTS[@]}" "$REMOTE_HOST" "REMOTE_ROOT='$REMOTE_ROOT' bash -s" > "$STAGING_DIR/remote_status.txt" <<'REMOTE'
set -euo pipefail

ROOT="${REMOTE_ROOT:-/path/to/privacy_distill_test_runs}"
cd "$ROOT"

echo "== date =="
date

echo
echo "== qstat =="
qstat -u "$USER" || true

echo
echo "== job ids =="
for job in 191247 191248 191249 191250 191251 191252 191253 191254; do
  echo "-- job $job --"
  qstat -j "$job" 2>/dev/null | sed -n '1,80p' || true
done

echo
echo "== dataset hashes =="
for ds in gpt-neo-fixed-20260419 pythia-fixed-20260419; do
  echo "-- $ds teacher_c2/teacher_c3 --"
  sha256sum "data/datasets/$ds/teacher_c2/"* "data/datasets/$ds/teacher_c3/"* 2>/dev/null || true
done

echo
echo "== model hashes =="
for tag in gpt-neo-2.7b gpt-neo-1.3b-local pythia-1.4b pythia-2.8b; do
  echo "-- $tag teachers --"
  sha256sum "outputs/$tag/teachers/c2/"*.safetensors "outputs/$tag/teachers/c3/"*.safetensors 2>/dev/null || true
  echo "-- $tag students --"
  sha256sum "outputs/$tag/students/c2/"*.safetensors "outputs/$tag/students/c3/"*.safetensors 2>/dev/null || true
done

echo
echo "== mia files =="
for tag in gpt-neo-2.7b gpt-neo-1.3b-local pythia-1.4b pythia-2.8b; do
  echo "-- $tag --"
  find "outputs/$tag/mia" -maxdepth 1 -type f 2>/dev/null | sort || true
done

echo
echo "== repair logs =="
find logs outputs/logs -maxdepth 2 -type f 2>/dev/null \
  \( -name '*191247*' -o -name '*191248*' -o -name '*191249*' -o -name '*191250*' \
     -o -name '*191251*' -o -name '*191252*' -o -name '*191253*' -o -name '*191254*' \
     -o -name '*c23-t2*' -o -name '*c23-s2*' -o -name '*c23-s3*' \) | sort || true
REMOTE

for tag in "${RUN_TAGS[@]}"; do
  mkdir -p "$STAGING_DIR/outputs/$tag"
  rsync "${RSYNC_OPTS[@]}" -e "$RSYNC_RSH" \
    "$REMOTE_HOST:$REMOTE_ROOT/outputs/$tag/mia/" \
    "$STAGING_DIR/outputs/$tag/mia/" || true
done

mkdir -p "$STAGING_DIR/logs"
for job in "${JOB_IDS[@]}"; do
  rsync "${RSYNC_OPTS[@]}" --include='*/' --include="*${job}*" --exclude='*' -e "$RSYNC_RSH" \
    "$REMOTE_HOST:$REMOTE_ROOT/logs/" \
    "$STAGING_DIR/logs/" || true
  rsync "${RSYNC_OPTS[@]}" --include='*/' --include="*${job}*" --exclude='*' -e "$RSYNC_RSH" \
    "$REMOTE_HOST:$REMOTE_ROOT/outputs/logs/" \
    "$STAGING_DIR/outputs/logs/" || true
done

for pattern in "${LOG_PATTERNS[@]}"; do
  rsync "${RSYNC_OPTS[@]}" --include='*/' --include="$pattern" --exclude='*' -e "$RSYNC_RSH" \
    "$REMOTE_HOST:$REMOTE_ROOT/logs/" \
    "$STAGING_DIR/logs/" || true
  rsync "${RSYNC_OPTS[@]}" --include='*/' --include="$pattern" --exclude='*' -e "$RSYNC_RSH" \
    "$REMOTE_HOST:$REMOTE_ROOT/outputs/logs/" \
    "$STAGING_DIR/outputs/logs/" || true
done

python3 - "$STAGING_DIR" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
tags = ["gpt-neo-2.7b", "gpt-neo-1.3b-local", "pythia-1.4b", "pythia-2.8b"]
names = [
    "c1_student.json",
    "c2_student.json",
    "c3_student.json",
    "c2_teacher.json",
    "c4_teacher.json",
    "utility_c2_student.json",
    "utility_c3_student.json",
    "utility_c2_student_holdout.json",
    "utility_c3_student_holdout.json",
    "utility_c2_teacher.json",
    "teacher_c1_vs_c2.json",
    "stats_bootstrap.json",
]

print("\n== local staged summary ==")
for tag in tags:
    mia = root / "outputs" / tag / "mia"
    print(f"-- {tag} --")
    for name in names:
        path = mia / name
        if not path.exists():
            print(f"missing {name}")
            continue
        try:
            data = json.loads(path.read_text())
        except Exception as exc:
            print(f"bad {name}: {exc}")
            continue
        if name.startswith("utility_"):
            print(f"{name}: loss={data.get('mean_loss')} ppl={data.get('perplexity')}")
        elif name.endswith("_student.json") or name.endswith("_teacher.json"):
            print(f"{name}: auroc={data.get('auroc')} auroc_doc={data.get('auroc_doc')}")
        else:
            print(f"{name}: ok")
PY

echo
echo "Fetch complete: $STAGING_DIR"
