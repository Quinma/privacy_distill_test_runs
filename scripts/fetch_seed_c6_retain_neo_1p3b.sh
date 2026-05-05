#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-ucacmqu@myriad.rc.ucl.ac.uk}"
REMOTE_ROOT="${REMOTE_ROOT:-/myriadfs/home/ucacmqu/privacy_distill_test_runs}"
RUN_TAG="${RUN_TAG:-gpt-neo-1.3b-local}"
NEO_SEEDS="${NEO_SEEDS:-17 19}"
STAGING_DIR="${STAGING_DIR:-$WORKSPACE_ROOT/exp/outputs/myriad_seed_c6_neo_20260505}"

mkdir -p "$STAGING_DIR/outputs/$RUN_TAG/seed_reps" "$STAGING_DIR/logs"
CONTROL_DIR="${TMPDIR:-/tmp}/codex-ssh-$USER"
mkdir -p "$CONTROL_DIR"
CONTROL_PATH="$CONTROL_DIR/$(echo "$REMOTE_HOST" | tr '@:/' '_').sock"
cleanup() {
  ssh -S "$CONTROL_PATH" -O exit "$REMOTE_HOST" >/dev/null 2>&1 || true
}
trap cleanup EXIT

ssh -M -S "$CONTROL_PATH" -fnNT "$REMOTE_HOST"
SSH_RSYNC=(rsync -av --partial --append-verify -e "ssh -S $CONTROL_PATH")
SSH_CMD=(ssh -S "$CONTROL_PATH" "$REMOTE_HOST")

"${SSH_CMD[@]}" "REMOTE_ROOT='$REMOTE_ROOT' RUN_TAG='$RUN_TAG' NEO_SEEDS='$NEO_SEEDS' bash -s" > "$STAGING_DIR/neo_seed_c6_remote_status.txt" <<'REMOTE'
set -euo pipefail
ROOT="${REMOTE_ROOT:-/myriadfs/home/ucacmqu/privacy_distill_test_runs}"
RUN_TAG="${RUN_TAG:-gpt-neo-1.3b-local}"
NEO_SEEDS="${NEO_SEEDS:-17 19}"
cd "$ROOT"
echo "== date =="; date
printf '\n== qstat ==\n'; qstat -u "$USER" || true
printf '\n== per-seed canonical outputs ==\n'
for seed in $NEO_SEEDS; do
  find "outputs/$RUN_TAG/seed_reps/seed_${seed}" -maxdepth 2 -type f \( -name 'mia_c6_deletion_attack_target_vs_retain*.json' -o -path '*/mia_retain/*_student_retain.json' -o -path '*/mia/c6_student.json' \) | sort || true
 done
printf '\n== summary files ==\n'
find "outputs/$RUN_TAG" -maxdepth 1 -type f -name 'seed_c6_deletion_attack_target_vs_retain*_summary.json' | sort || true
printf '\n== logs ==\n'
find logs -maxdepth 1 -type f -name '*n13-c6sd*' | sort || true
REMOTE

for seed in $NEO_SEEDS; do
  mkdir -p "$STAGING_DIR/outputs/$RUN_TAG/seed_reps/seed_${seed}"
  "${SSH_RSYNC[@]}" \
    "$REMOTE_HOST:$REMOTE_ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/mia_c6_deletion_attack_target_vs_retain.json" \
    "$STAGING_DIR/outputs/$RUN_TAG/seed_reps/seed_${seed}/" || true
  "${SSH_RSYNC[@]}" \
    "$REMOTE_HOST:$REMOTE_ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/mia_c6_deletion_attack_target_vs_retain_c1ref.json" \
    "$STAGING_DIR/outputs/$RUN_TAG/seed_reps/seed_${seed}/" || true
  mkdir -p "$STAGING_DIR/outputs/$RUN_TAG/seed_reps/seed_${seed}/mia_retain"
  "${SSH_RSYNC[@]}" \
    "$REMOTE_HOST:$REMOTE_ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/mia_retain/" \
    "$STAGING_DIR/outputs/$RUN_TAG/seed_reps/seed_${seed}/mia_retain/" || true
  mkdir -p "$STAGING_DIR/outputs/$RUN_TAG/seed_reps/seed_${seed}/mia"
  "${SSH_RSYNC[@]}" \
    "$REMOTE_HOST:$REMOTE_ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/mia/c6_student.json" \
    "$STAGING_DIR/outputs/$RUN_TAG/seed_reps/seed_${seed}/mia/" || true
done

"${SSH_RSYNC[@]}" \
  "$REMOTE_HOST:$REMOTE_ROOT/outputs/$RUN_TAG/seed_c6_deletion_attack_target_vs_retain_summary.json" \
  "$STAGING_DIR/outputs/$RUN_TAG/" || true
"${SSH_RSYNC[@]}" \
  "$REMOTE_HOST:$REMOTE_ROOT/outputs/$RUN_TAG/seed_c6_deletion_attack_target_vs_retain_c1ref_summary.json" \
  "$STAGING_DIR/outputs/$RUN_TAG/" || true
"${SSH_RSYNC[@]}" --include='*/' --include='*n13-c6sd*' --exclude='*' \
  "$REMOTE_HOST:$REMOTE_ROOT/logs/" \
  "$STAGING_DIR/logs/" || true

echo "Fetch complete: $STAGING_DIR"
