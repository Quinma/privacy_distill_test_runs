#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXP_ROOT="$WORKSPACE_ROOT/exp"
LOCAL_ROOT="$WORKSPACE_ROOT/local_repo"

REMOTE_HOST="${REMOTE_HOST:-ucacmqu@myriad.rc.ucl.ac.uk}"
REMOTE_ROOT="${REMOTE_ROOT:-/myriadfs/home/ucacmqu/privacy_distill_test_runs}"
FAMILIES="${FAMILIES:-pythia neo}"
PYTHIA_SEEDS="${PYTHIA_SEEDS:-13 17 19}"
NEO_SEEDS="${NEO_SEEDS:-17 19}"
NEO_RUN_TAG="${NEO_RUN_TAG:-gpt-neo-1.3b-local}"

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

remote_mkdir_seed_parents() {
  local base="$1"
  shift
  local cmd="mkdir -p '$REMOTE_ROOT/scripts'"
  for seed in "$@"; do
    cmd="$cmd '$REMOTE_ROOT/$base/seed_reps/seed_${seed}/teachers' '$REMOTE_ROOT/$base/seed_reps/seed_${seed}/students' '$REMOTE_ROOT/$base/seed_reps/seed_${seed}/mia' '$REMOTE_ROOT/$base/seed_reps/seed_${seed}/mia_retain'"
  done
  "${SSH_CMD[@]}" "$cmd"
}

sync_common_scripts() {
  echo "Syncing shared seeded-placebo scripts"
  "${SSH_RSYNC[@]}" \
    "$WORKSPACE_ROOT/cluster_repo/scripts/run_seed_placebo_c1_retain.sh" \
    "$WORKSPACE_ROOT/cluster_repo/scripts/eval_company_losses.py" \
    "$WORKSPACE_ROOT/cluster_repo/scripts/finalize_placebo_attack.py" \
    "$REMOTE_HOST:$REMOTE_ROOT/scripts/"
}

sync_pythia_seed_refs() {
  echo "Syncing Pythia seeded C1 artifacts: $PYTHIA_SEEDS"
  remote_mkdir_seed_parents "outputs/pythia-1.4b" $PYTHIA_SEEDS
  "${SSH_CMD[@]}" "mkdir -p '$REMOTE_ROOT/data/datasets/pythia-1.4b' '$REMOTE_ROOT/data'"
  "${SSH_RSYNC[@]}" \
    "$EXP_ROOT/data/datasets/pythia-1.4b/splits.json" \
    "$EXP_ROOT/data/datasets/pythia-1.4b/holdout_map.json" \
    "$REMOTE_HOST:$REMOTE_ROOT/data/datasets/pythia-1.4b/"
  "${SSH_RSYNC[@]}" \
    "$EXP_ROOT/data/datasets/pythia-1.4b/random_forget_train" \
    "$REMOTE_HOST:$REMOTE_ROOT/data/datasets/pythia-1.4b/"
  "${SSH_RSYNC[@]}" \
    "$EXP_ROOT/data/bean_counter_stats.jsonl" \
    "$EXP_ROOT/data/sec_index_10k.jsonl" \
    "$REMOTE_HOST:$REMOTE_ROOT/data/"
  for seed in $PYTHIA_SEEDS; do
    "${SSH_CMD[@]}" "mkdir -p '$REMOTE_ROOT/outputs/pythia-1.4b/seed_reps/seed_${seed}/c5r_a03'"
    if [[ -f "$EXP_ROOT/outputs/pythia-1.4b/seed_reps/seed_${seed}/c5r_a03/c5r_forget_ciks.json" ]]; then
      "${SSH_RSYNC[@]}" \
        "$EXP_ROOT/outputs/pythia-1.4b/seed_reps/seed_${seed}/c5r_a03/c5r_forget_ciks.json" \
        "$REMOTE_HOST:$REMOTE_ROOT/outputs/pythia-1.4b/seed_reps/seed_${seed}/c5r_a03/"
    fi
    "${SSH_RSYNC[@]}" \
      "$EXP_ROOT/outputs/pythia-1.4b/seed_reps/seed_${seed}/teachers/c1" \
      "$REMOTE_HOST:$REMOTE_ROOT/outputs/pythia-1.4b/seed_reps/seed_${seed}/teachers/"
    "${SSH_RSYNC[@]}" \
      "$EXP_ROOT/outputs/pythia-1.4b/seed_reps/seed_${seed}/students/c1" \
      "$REMOTE_HOST:$REMOTE_ROOT/outputs/pythia-1.4b/seed_reps/seed_${seed}/students/"
    "${SSH_RSYNC[@]}" \
      "$EXP_ROOT/outputs/pythia-1.4b/seed_reps/seed_${seed}/mia/c1_student.json" \
      "$REMOTE_HOST:$REMOTE_ROOT/outputs/pythia-1.4b/seed_reps/seed_${seed}/mia/"
    "${SSH_RSYNC[@]}" \
      "$EXP_ROOT/outputs/pythia-1.4b/seed_reps/seed_${seed}/mia_retain/c1_student_retain.json" \
      "$REMOTE_HOST:$REMOTE_ROOT/outputs/pythia-1.4b/seed_reps/seed_${seed}/mia_retain/"
  done
}

sync_neo_seed_refs() {
  echo "Syncing Neo seeded C1 artifacts: $NEO_SEEDS"
  remote_mkdir_seed_parents "outputs/$NEO_RUN_TAG" $NEO_SEEDS
  "${SSH_CMD[@]}" "mkdir -p '$REMOTE_ROOT/data/datasets/gpt-neo-fixed-20260419' '$REMOTE_ROOT/data'"
  "${SSH_RSYNC[@]}" \
    "$LOCAL_ROOT/data/datasets/gpt-neo-fixed-20260419/splits.json" \
    "$LOCAL_ROOT/data/datasets/gpt-neo-fixed-20260419/holdout_map.json" \
    "$REMOTE_HOST:$REMOTE_ROOT/data/datasets/gpt-neo-fixed-20260419/"
  "${SSH_RSYNC[@]}" \
    "$LOCAL_ROOT/data/datasets/gpt-neo-fixed-20260419/random_forget_train" \
    "$REMOTE_HOST:$REMOTE_ROOT/data/datasets/gpt-neo-fixed-20260419/"
  "${SSH_RSYNC[@]}" \
    "$LOCAL_ROOT/data/bean_counter_stats.jsonl" \
    "$LOCAL_ROOT/data/sec_index_10k.jsonl" \
    "$REMOTE_HOST:$REMOTE_ROOT/data/"
  for seed in $NEO_SEEDS; do
    "${SSH_CMD[@]}" "mkdir -p '$REMOTE_ROOT/outputs/$NEO_RUN_TAG/seed_reps/seed_${seed}/c5r_a03'"
    if [[ -f "$LOCAL_ROOT/outputs/$NEO_RUN_TAG/seed_reps/seed_${seed}/c5r_a03/c5r_forget_ciks.json" ]]; then
      "${SSH_RSYNC[@]}" \
        "$LOCAL_ROOT/outputs/$NEO_RUN_TAG/seed_reps/seed_${seed}/c5r_a03/c5r_forget_ciks.json" \
        "$REMOTE_HOST:$REMOTE_ROOT/outputs/$NEO_RUN_TAG/seed_reps/seed_${seed}/c5r_a03/"
    fi
    "${SSH_RSYNC[@]}" \
      "$LOCAL_ROOT/outputs/$NEO_RUN_TAG/seed_reps/seed_${seed}/teachers/c1" \
      "$REMOTE_HOST:$REMOTE_ROOT/outputs/$NEO_RUN_TAG/seed_reps/seed_${seed}/teachers/"
    "${SSH_RSYNC[@]}" \
      "$LOCAL_ROOT/outputs/$NEO_RUN_TAG/seed_reps/seed_${seed}/students/c1" \
      "$REMOTE_HOST:$REMOTE_ROOT/outputs/$NEO_RUN_TAG/seed_reps/seed_${seed}/students/"
    if [[ -f "$LOCAL_ROOT/outputs/$NEO_RUN_TAG/seed_reps/seed_${seed}/mia/c1_student.json" ]]; then
      "${SSH_RSYNC[@]}" \
        "$LOCAL_ROOT/outputs/$NEO_RUN_TAG/seed_reps/seed_${seed}/mia/c1_student.json" \
        "$REMOTE_HOST:$REMOTE_ROOT/outputs/$NEO_RUN_TAG/seed_reps/seed_${seed}/mia/"
    fi
    if [[ -f "$LOCAL_ROOT/outputs/$NEO_RUN_TAG/seed_reps/seed_${seed}/mia_retain/c1_student_retain.json" ]]; then
      "${SSH_RSYNC[@]}" \
        "$LOCAL_ROOT/outputs/$NEO_RUN_TAG/seed_reps/seed_${seed}/mia_retain/c1_student_retain.json" \
        "$REMOTE_HOST:$REMOTE_ROOT/outputs/$NEO_RUN_TAG/seed_reps/seed_${seed}/mia_retain/"
    fi
  done
}

submit_pythia() {
  echo "Submitting Pythia seeded placebo job"
  "${SSH_CMD[@]}" "REMOTE_ROOT='$REMOTE_ROOT' PYTHIA_SEEDS='$PYTHIA_SEEDS' bash -s" <<'REMOTE'
set -euo pipefail

ROOT="${REMOTE_ROOT:-/myriadfs/home/ucacmqu/privacy_distill_test_runs}"
PYTHIA_SEEDS="${PYTHIA_SEEDS:-13 17 19}"
cd "$ROOT"

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: missing required directory: $path" >&2
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
require_dir "$ROOT/data/datasets/pythia-1.4b/teacher_c2"
require_dir "$ROOT/data/datasets/eval_retain_holdout"
require_file "$ROOT/data/datasets/pythia-1.4b/splits.json"
require_file "$ROOT/data/datasets/pythia-1.4b/holdout_map.json"
require_file "$ROOT/data/sec_index_10k.jsonl"
require_file "$ROOT/data/bean_counter_stats.jsonl"
require_dir "$ROOT/scripts"
require_dir "$ROOT/cluster"

for seed in $PYTHIA_SEEDS; do
  require_model_dir "$ROOT/outputs/pythia-1.4b/seed_reps/seed_${seed}/teachers/c1"
  require_model_dir "$ROOT/outputs/pythia-1.4b/seed_reps/seed_${seed}/students/c1"
  require_file "$ROOT/outputs/pythia-1.4b/seed_reps/seed_${seed}/mia/c1_student.json"
  require_file "$ROOT/outputs/pythia-1.4b/seed_reps/seed_${seed}/mia_retain/c1_student_retain.json"
done

qsub -N p14-plc1sd \
  -l h_rt=48:00:00 -l mem=8G -l gpu=4 \
  -pe smp 8 -ac allow=L \
  -v REPO_ROOT="$ROOT",FAMILIES=pythia,PYTHIA_SEEDS="$PYTHIA_SEEDS",PYTHIA_DISTILL_OPTIM=adamw_torch,RUN_CMD=./scripts/run_seed_placebo_c1_retain.sh \
  cluster/qsub_run.sh
REMOTE
}

submit_neo() {
  echo "Submitting Neo seeded placebo job"
  "${SSH_CMD[@]}" "REMOTE_ROOT='$REMOTE_ROOT' NEO_SEEDS='$NEO_SEEDS' NEO_RUN_TAG='$NEO_RUN_TAG' bash -s" <<'REMOTE'
set -euo pipefail

ROOT="${REMOTE_ROOT:-/myriadfs/home/ucacmqu/privacy_distill_test_runs}"
NEO_SEEDS="${NEO_SEEDS:-17 19}"
NEO_RUN_TAG="${NEO_RUN_TAG:-gpt-neo-1.3b-local}"
cd "$ROOT"

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: missing required directory: $path" >&2
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

require_model_dir() {
  local path="$1"
  if [[ ! -f "$path/model.safetensors" && ! -f "$path/model.safetensors.index.json" && ! -f "$path/pytorch_model.bin" && ! -f "$path/model-00001-of-00002.safetensors" && ! -f "$path/model-00001-of-00003.safetensors" ]]; then
    echo "ERROR: missing model artifacts in $path" >&2
    exit 1
  fi
}

require_dir "$ROOT/data/datasets/gpt-neo-fixed-20260419/distill"
require_dir "$ROOT/data/datasets/gpt-neo-fixed-20260419/eval_target_holdout"
require_dir "$ROOT/data/datasets/gpt-neo-fixed-20260419/eval_nonmember"
require_dir "$ROOT/data/datasets/gpt-neo-fixed-20260419/teacher_c2"
require_dir "$ROOT/data/datasets/eval_retain_holdout"
require_file "$ROOT/data/datasets/gpt-neo-fixed-20260419/splits.json"
require_file "$ROOT/data/datasets/gpt-neo-fixed-20260419/holdout_map.json"
require_file "$ROOT/data/sec_index_10k.jsonl"
require_file "$ROOT/data/bean_counter_stats.jsonl"
require_dir "$ROOT/scripts"
require_dir "$ROOT/cluster"

for seed in $NEO_SEEDS; do
  require_model_dir "$ROOT/outputs/$NEO_RUN_TAG/seed_reps/seed_${seed}/teachers/c1"
  require_model_dir "$ROOT/outputs/$NEO_RUN_TAG/seed_reps/seed_${seed}/students/c1"
  require_file "$ROOT/outputs/$NEO_RUN_TAG/seed_reps/seed_${seed}/mia/c1_student.json"
done

qsub -N n13-plc1sd \
  -l h_rt=48:00:00 -l mem=8G -l gpu=4 \
  -pe smp 8 -ac allow=L \
  -v REPO_ROOT="$ROOT",FAMILIES=neo,NEO_SEEDS="$NEO_SEEDS",NEO_DISTILL_OPTIM=adamw_torch,RUN_CMD=./scripts/run_seed_placebo_c1_retain.sh \
  cluster/qsub_run.sh
REMOTE
}

sync_common_scripts

for family in $FAMILIES; do
  case "$family" in
    pythia) sync_pythia_seed_refs ;;
    neo) sync_neo_seed_refs ;;
    *) echo "ERROR: unknown family '$family' (expected pythia and/or neo)" >&2; exit 1 ;;
  esac
done

for family in $FAMILIES; do
  case "$family" in
    pythia) submit_pythia ;;
    neo) submit_neo ;;
  esac
done

echo "Seeded placebo sync and submit complete."
