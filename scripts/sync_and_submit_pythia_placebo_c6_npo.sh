#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-ucacmqu@myriad.rc.ucl.ac.uk}"
REMOTE_ROOT="${REMOTE_ROOT:-/myriadfs/home/ucacmqu/privacy_distill_test_runs}"
RUN_TAG="${RUN_TAG:-pythia-1.4b-placebo-npo-s13}"

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
SSH_CMD=(ssh -S "$CONTROL_PATH" "$REMOTE_HOST")

"${SSH_CMD[@]}" "mkdir -p \
  '$REMOTE_ROOT/scripts' \
  '$REMOTE_ROOT/data/datasets/pythia-1.4b' \
  '$REMOTE_ROOT/outputs/$RUN_TAG/reference_eval' \
  '$REMOTE_ROOT/outputs/pythia-1.4b/teachers' \
  '$REMOTE_ROOT/outputs/pythia-1.4b/students'"

echo "Syncing placebo scripts"
"${SSH_RSYNC[@]}" \
  "$WORKSPACE_ROOT/exp/scripts/run_c6_npo.sh" \
  "$WORKSPACE_ROOT/cluster_repo/scripts/run_placebo_c6_npo_1p4b.sh" \
  "$WORKSPACE_ROOT/cluster_repo/scripts/fetch_pythia_placebo_c6_npo.sh" \
  "$REMOTE_HOST:$REMOTE_ROOT/scripts/"

echo "Syncing random-forget dataset"
"${SSH_RSYNC[@]}" \
  "$WORKSPACE_ROOT/exp/data/datasets/pythia-1.4b/random_forget_train" \
  "$WORKSPACE_ROOT/exp/data/datasets/pythia-1.4b/random_forget_meta.json" \
  "$REMOTE_HOST:$REMOTE_ROOT/data/datasets/pythia-1.4b/"

echo "Syncing reference eval JSONs"
"${SSH_RSYNC[@]}" \
  "$WORKSPACE_ROOT/exp/outputs/pythia-1.4b/mia/c1_student.json" \
  "$WORKSPACE_ROOT/exp/outputs/pythia-1.4b/mia/c3_student.json" \
  "$WORKSPACE_ROOT/exp/outputs/pythia-1.4b/mia_retain/c1_student_retain.json" \
  "$WORKSPACE_ROOT/exp/outputs/pythia-1.4b/mia_retain/c3_student_retain.json" \
  "$REMOTE_HOST:$REMOTE_ROOT/outputs/$RUN_TAG/reference_eval/"

echo "Syncing required Pythia model artifacts"
"${SSH_RSYNC[@]}" \
  "$WORKSPACE_ROOT/exp/outputs/pythia-1.4b/teachers/c1" \
  "$REMOTE_HOST:$REMOTE_ROOT/outputs/pythia-1.4b/teachers/"
"${SSH_RSYNC[@]}" \
  "$WORKSPACE_ROOT/exp/outputs/pythia-1.4b/students/c1" \
  "$WORKSPACE_ROOT/exp/outputs/pythia-1.4b/students/c3" \
  "$REMOTE_HOST:$REMOTE_ROOT/outputs/pythia-1.4b/students/"

echo "Submitting placebo job"
"${SSH_CMD[@]}" "REMOTE_ROOT='$REMOTE_ROOT' RUN_TAG='$RUN_TAG' bash -s" <<'REMOTE'
set -euo pipefail

ROOT="${REMOTE_ROOT:-/myriadfs/home/ucacmqu/privacy_distill_test_runs}"
RUN_TAG="${RUN_TAG:-pythia-1.4b-placebo-npo-s13}"
cd "$ROOT"

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: missing required directory: $path" >&2
    exit 1
  fi
}

require_model_dir() {
  local path="$1"
  if [[ ! -f "$path/model.safetensors" && ! -f "$path/model.safetensors.index.json" && ! -f "$path/pytorch_model.bin" && ! -f "$path/model-00001-of-00002.safetensors" && ! -f "$path/model-00001-of-00003.safetensors" ]]; then
    echo "ERROR: missing model artifacts in $path" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required file: $path" >&2
    exit 1
  fi
}

require_dir "$ROOT/data/datasets/pythia-1.4b/distill"
require_dir "$ROOT/data/datasets/pythia-1.4b/eval_target_holdout"
require_dir "$ROOT/data/datasets/pythia-1.4b/eval_nonmember"
require_dir "$ROOT/data/datasets/eval_retain_holdout"
require_dir "$ROOT/data/datasets/pythia-1.4b/random_forget_train"
require_dir "$ROOT/data/datasets/pythia-1.4b/teacher_c2"
require_model_dir "$ROOT/outputs/pythia-1.4b/teachers/c1"
require_model_dir "$ROOT/outputs/pythia-1.4b/students/c1"
require_model_dir "$ROOT/outputs/pythia-1.4b/students/c3"
require_file "$ROOT/outputs/$RUN_TAG/reference_eval/c1_student.json"
require_file "$ROOT/outputs/$RUN_TAG/reference_eval/c1_student_retain.json"
require_file "$ROOT/outputs/$RUN_TAG/reference_eval/c3_student.json"
require_file "$ROOT/outputs/$RUN_TAG/reference_eval/c3_student_retain.json"
require_dir "$ROOT/scripts"
require_dir "$ROOT/cluster"

qsub -N p14-placebo-c6 \
  -l h_rt=30:00:00 -l mem=8G -l gpu=4 \
  -pe smp 8 -ac allow=L \
  -v REPO_ROOT="$ROOT",RUN_TAG="$RUN_TAG",RUN_CMD=./scripts/run_placebo_c6_npo_1p4b.sh \
  cluster/qsub_run.sh
REMOTE

echo "Sync and submit complete."
