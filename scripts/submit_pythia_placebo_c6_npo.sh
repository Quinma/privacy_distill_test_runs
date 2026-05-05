#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-ucacmqu@myriad.rc.ucl.ac.uk}"
REMOTE_ROOT="${REMOTE_ROOT:-/myriadfs/home/ucacmqu/privacy_distill_test_runs}"

ssh "$REMOTE_HOST" "REMOTE_ROOT='$REMOTE_ROOT' bash -s" <<'REMOTE'
set -euo pipefail

ROOT="${REMOTE_ROOT:-/myriadfs/home/ucacmqu/privacy_distill_test_runs}"
RUN_TAG="${RUN_TAG:-pythia-1.4b-placebo-npo-s13}"
REF_EVAL_DIR="$ROOT/outputs/$RUN_TAG/reference_eval"
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

require_dir "$ROOT/data/datasets/pythia-1.4b/distill"
require_dir "$ROOT/data/datasets/pythia-1.4b/eval_target_holdout"
require_dir "$ROOT/data/datasets/pythia-1.4b/eval_nonmember"
require_dir "$ROOT/data/datasets/eval_retain_holdout"
require_dir "$ROOT/data/datasets/pythia-1.4b/random_forget_train"
require_dir "$ROOT/data/datasets/pythia-1.4b/teacher_c2"
require_model_dir "$ROOT/outputs/pythia-1.4b/teachers/c1"

if [[ ! -f "$REF_EVAL_DIR/c1_student_target.json" || ! -f "$REF_EVAL_DIR/c1_student_retain.json" ]]; then
  require_model_dir "$ROOT/outputs/pythia-1.4b/students/c1"
fi

if [[ ! -f "$REF_EVAL_DIR/c3_student_target.json" || ! -f "$REF_EVAL_DIR/c3_student_retain.json" ]]; then
  require_model_dir "$ROOT/outputs/pythia-1.4b/students/c3"
fi

require_dir "$ROOT/scripts"
require_dir "$ROOT/cluster"

qsub -N p14-placebo-c6 \
  -l h_rt=30:00:00 -l mem=8G -l gpu=4 \
  -pe smp 8 -ac allow=L \
  -v REPO_ROOT="$ROOT",RUN_CMD=./scripts/run_placebo_c6_npo_1p4b.sh \
  cluster/qsub_run.sh
REMOTE
