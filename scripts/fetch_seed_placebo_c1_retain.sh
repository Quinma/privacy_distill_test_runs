#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-ucacmqu@myriad.rc.ucl.ac.uk}"
REMOTE_ROOT="${REMOTE_ROOT:-/myriadfs/home/ucacmqu/privacy_distill_test_runs}"
FAMILIES="${FAMILIES:-pythia neo}"
PYTHIA_SEEDS="${PYTHIA_SEEDS:-13 17 19}"
NEO_SEEDS="${NEO_SEEDS:-17 19}"
NEO_RUN_TAG="${NEO_RUN_TAG:-gpt-neo-1.3b-local}"
STAGING_DIR="${STAGING_DIR:-$WORKSPACE_ROOT/exp/outputs/myriad_seed_placebo_c1_20260428}"

mkdir -p "$STAGING_DIR/outputs" "$STAGING_DIR/logs"

CONTROL_DIR="${TMPDIR:-/tmp}/codex-ssh-$USER"
mkdir -p "$CONTROL_DIR"
CONTROL_PATH="$CONTROL_DIR/$(echo "$REMOTE_HOST" | tr '@:/' '_').sock"

cleanup() {
  ssh -S "$CONTROL_PATH" -O exit "$REMOTE_HOST" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "Opening shared SSH connection to $REMOTE_HOST"
ssh -M -S "$CONTROL_PATH" -fnNT "$REMOTE_HOST"

SSH_RSYNC=(rsync -av --partial --append-verify -e "ssh -S $CONTROL_PATH")

fetch_pythia() {
  mkdir -p "$STAGING_DIR/outputs/pythia-1.4b/seed_reps"
  for seed in $PYTHIA_SEEDS; do
    mkdir -p "$STAGING_DIR/outputs/pythia-1.4b/seed_reps/seed_${seed}"
    "${SSH_RSYNC[@]}" \
      "$REMOTE_HOST:$REMOTE_ROOT/outputs/pythia-1.4b/seed_reps/seed_${seed}/placebo_c6_c1_retain/" \
      "$STAGING_DIR/outputs/pythia-1.4b/seed_reps/seed_${seed}/placebo_c6_c1_retain/" || true
  done
  "${SSH_RSYNC[@]}" \
    "$REMOTE_HOST:$REMOTE_ROOT/outputs/pythia-1.4b/seed_c6_placebo_deletion_attack_target_vs_retain_c1ref_summary.json" \
    "$STAGING_DIR/outputs/pythia-1.4b/" || true
  "${SSH_RSYNC[@]}" --include='*/' --include='*p14-plc1sd*' --exclude='*' \
    "$REMOTE_HOST:$REMOTE_ROOT/logs/" \
    "$STAGING_DIR/logs/" || true
}

fetch_neo() {
  mkdir -p "$STAGING_DIR/outputs/$NEO_RUN_TAG/seed_reps"
  for seed in $NEO_SEEDS; do
    mkdir -p "$STAGING_DIR/outputs/$NEO_RUN_TAG/seed_reps/seed_${seed}"
    "${SSH_RSYNC[@]}" \
      "$REMOTE_HOST:$REMOTE_ROOT/outputs/$NEO_RUN_TAG/seed_reps/seed_${seed}/placebo_c6_c1_retain/" \
      "$STAGING_DIR/outputs/$NEO_RUN_TAG/seed_reps/seed_${seed}/placebo_c6_c1_retain/" || true
  done
  "${SSH_RSYNC[@]}" \
    "$REMOTE_HOST:$REMOTE_ROOT/outputs/$NEO_RUN_TAG/seed_c6_placebo_deletion_attack_target_vs_retain_c1ref_summary.json" \
    "$STAGING_DIR/outputs/$NEO_RUN_TAG/" || true
  "${SSH_RSYNC[@]}" --include='*/' --include='*n13-plc1sd*' --exclude='*' \
    "$REMOTE_HOST:$REMOTE_ROOT/logs/" \
    "$STAGING_DIR/logs/" || true
}

for family in $FAMILIES; do
  case "$family" in
    pythia) fetch_pythia ;;
    neo) fetch_neo ;;
    *) echo "ERROR: unknown family '$family' (expected pythia and/or neo)" >&2; exit 1 ;;
  esac
done

echo "Fetch complete: $STAGING_DIR"
