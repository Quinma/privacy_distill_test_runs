#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
LOCAL_ROOT="$WORKSPACE_ROOT/local_repo"
EXP_ROOT="$WORKSPACE_ROOT/exp"

REMOTE_HOST="${REMOTE_HOST:-user@cluster.example.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/path/to/privacy_distill_test_runs}"
NEO_SEEDS="${NEO_SEEDS:-17 19}"
RUN_TAG="${RUN_TAG:-gpt-neo-1.3b-local}"
DATASET_TAG="${DATASET_TAG:-gpt-neo-fixed-20260419}"

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

remote_mkdir_seed_parents() {
  local cmd="mkdir -p '$REMOTE_ROOT/scripts' '$REMOTE_ROOT/data/datasets/$DATASET_TAG' '$REMOTE_ROOT/data/datasets/eval_retain_holdout' '$REMOTE_ROOT/logs'"
  for seed in $NEO_SEEDS; do
    cmd="$cmd '$REMOTE_ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/teachers' '$REMOTE_ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/students' '$REMOTE_ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/mia' '$REMOTE_ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/mia_retain'"
  done
  "${SSH_CMD[@]}" "$cmd"
}

sync_scripts() {
  "${SSH_RSYNC[@]}" \
    "$WORKSPACE_ROOT/cluster_repo/scripts/run_seed_c6_npo_neo_1p3b.sh" \
    "$WORKSPACE_ROOT/cluster_repo/scripts/run_c6_npo_neo_1p3b.sh" \
    "$WORKSPACE_ROOT/cluster_repo/scripts/run_seed_c6_retain_attack_neo_1p3b.sh" \
    "$WORKSPACE_ROOT/cluster_repo/scripts/eval_company_losses.py" \
    "$WORKSPACE_ROOT/cluster_repo/scripts/finalize_deletion_attack.py" \
    "$REMOTE_HOST:$REMOTE_ROOT/scripts/"
}

sync_datasets() {
  "${SSH_RSYNC[@]}"     "$LOCAL_ROOT/data/datasets/$DATASET_TAG/"     "$REMOTE_HOST:$REMOTE_ROOT/data/datasets/$DATASET_TAG/"
  if [[ -d "$LOCAL_ROOT/data/datasets/gpt-neo-1.3b-local/target_train" ]]; then
    "${SSH_RSYNC[@]}"       "$LOCAL_ROOT/data/datasets/gpt-neo-1.3b-local/target_train/"       "$REMOTE_HOST:$REMOTE_ROOT/data/datasets/$DATASET_TAG/target_train/"
  fi
  "${SSH_RSYNC[@]}"     "$EXP_ROOT/data/datasets/eval_retain_holdout/"     "$REMOTE_HOST:$REMOTE_ROOT/data/datasets/eval_retain_holdout/"
}

sync_seed_inputs() {
  for seed in $NEO_SEEDS; do
    "${SSH_RSYNC[@]}" \
      "$LOCAL_ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/teachers/c1" \
      "$REMOTE_HOST:$REMOTE_ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/teachers/"
    "${SSH_RSYNC[@]}" \
      "$LOCAL_ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/students/c1" \
      "$REMOTE_HOST:$REMOTE_ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/students/"
    "${SSH_RSYNC[@]}" \
      "$LOCAL_ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/students/c3" \
      "$REMOTE_HOST:$REMOTE_ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/students/"
    "${SSH_RSYNC[@]}" \
      "$LOCAL_ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/mia/c1_student.json" \
      "$REMOTE_HOST:$REMOTE_ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/mia/"
    "${SSH_RSYNC[@]}" \
      "$LOCAL_ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/mia/c3_student.json" \
      "$REMOTE_HOST:$REMOTE_ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/mia/"
  done
}

submit_job() {
  "${SSH_CMD[@]}" "REMOTE_ROOT='$REMOTE_ROOT' NEO_SEEDS='$NEO_SEEDS' RUN_TAG='$RUN_TAG' DATASET_TAG='$DATASET_TAG' bash -s" <<'REMOTE'
set -euo pipefail
ROOT="${REMOTE_ROOT:-/path/to/privacy_distill_test_runs}"
NEO_SEEDS="${NEO_SEEDS:-17 19}"
RUN_TAG="${RUN_TAG:-gpt-neo-1.3b-local}"
DATASET_TAG="${DATASET_TAG:-gpt-neo-fixed-20260419}"
cd "$ROOT"

require_dir() { [[ -d "$1" ]] || { echo "ERROR: missing required directory: $1" >&2; exit 1; }; }
require_file() { [[ -f "$1" ]] || { echo "ERROR: missing required file: $1" >&2; exit 1; }; }
require_model_dir() {
  local path="$1"
  if [[ ! -f "$path/model.safetensors" && ! -f "$path/model.safetensors.index.json" && ! -f "$path/pytorch_model.bin" && ! -f "$path/model-00001-of-00002.safetensors" && ! -f "$path/model-00001-of-00003.safetensors" ]]; then
    echo "ERROR: missing model artifacts in $path" >&2
    exit 1
  fi
}

require_dir "$ROOT/data/datasets/$DATASET_TAG/distill"
require_dir "$ROOT/data/datasets/$DATASET_TAG/target_train"
require_dir "$ROOT/data/datasets/$DATASET_TAG/teacher_c2"
require_dir "$ROOT/data/datasets/$DATASET_TAG/eval_target_holdout"
require_dir "$ROOT/data/datasets/$DATASET_TAG/eval_nonmember"
require_dir "$ROOT/data/datasets/eval_retain_holdout"
require_file "$ROOT/data/datasets/$DATASET_TAG/holdout_map.json"
for seed in $NEO_SEEDS; do
  require_model_dir "$ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/teachers/c1"
  require_model_dir "$ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/students/c1"
  require_model_dir "$ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/students/c3"
  require_file "$ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/mia/c1_student.json"
  require_file "$ROOT/outputs/$RUN_TAG/seed_reps/seed_${seed}/mia/c3_student.json"
done

qsub -N n13-c6sd \
  -l h_rt=48:00:00 -l mem=8G -l gpu=4 \
  -pe smp 8 -ac allow=L \
  -v REPO_ROOT="$ROOT",RUN_TAG="$RUN_TAG",DATASETS_DIR="$ROOT/data/datasets/$DATASET_TAG",SEED_ROOT="$ROOT/outputs/$RUN_TAG/seed_reps",SEEDS="$NEO_SEEDS",RETAIN_HOLDOUT_DATA="$ROOT/data/datasets/eval_retain_holdout",RUN_CMD=./scripts/run_seed_c6_retain_attack_neo_1p3b.sh \
  cluster/qsub_run.sh
REMOTE
}

remote_mkdir_seed_parents
sync_scripts
sync_datasets
sync_seed_inputs
submit_job

echo "Seeded canonical Neo sync and submit complete."
