#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-user@cluster.example.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/path/to/privacy_distill_test_runs}"
RUN_TAG="${RUN_TAG:-pythia-1.4b-placebo-npo-s13}"

ssh "$REMOTE_HOST" "mkdir -p '$REMOTE_ROOT/scripts' '$REMOTE_ROOT/data/datasets/pythia-1.4b' '$REMOTE_ROOT/outputs/$RUN_TAG/reference_eval'"

rsync -av --partial --append-verify \
  "$WORKSPACE_ROOT/exp/scripts/run_c6_npo.sh" \
  "$WORKSPACE_ROOT/cluster_repo/scripts/run_placebo_c6_npo_1p4b.sh" \
  "$WORKSPACE_ROOT/cluster_repo/scripts/submit_pythia_placebo_c6_npo.sh" \
  "$WORKSPACE_ROOT/cluster_repo/scripts/fetch_pythia_placebo_c6_npo.sh" \
  "$REMOTE_HOST:$REMOTE_ROOT/scripts/"

rsync -av --partial --append-verify \
  "$WORKSPACE_ROOT/exp/data/datasets/pythia-1.4b/random_forget_train" \
  "$WORKSPACE_ROOT/exp/data/datasets/pythia-1.4b/random_forget_meta.json" \
  "$REMOTE_HOST:$REMOTE_ROOT/data/datasets/pythia-1.4b/"

rsync -av --partial --append-verify \
  "$WORKSPACE_ROOT/exp/outputs/pythia-1.4b/mia/c1_student.json" \
  "$WORKSPACE_ROOT/exp/outputs/pythia-1.4b/mia/c3_student.json" \
  "$WORKSPACE_ROOT/exp/outputs/pythia-1.4b/mia_retain/c1_student_retain.json" \
  "$WORKSPACE_ROOT/exp/outputs/pythia-1.4b/mia_retain/c3_student_retain.json" \
  "$REMOTE_HOST:$REMOTE_ROOT/outputs/$RUN_TAG/reference_eval/"

echo "Synced placebo C6 scripts and random-forget data to $REMOTE_HOST:$REMOTE_ROOT"
