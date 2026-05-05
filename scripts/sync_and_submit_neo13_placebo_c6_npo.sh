#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOCAL_ROOT="$WORKSPACE_ROOT/local_repo"

REMOTE_HOST="${REMOTE_HOST:-ucacmqu@myriad.rc.ucl.ac.uk}"
REMOTE_ROOT="${REMOTE_ROOT:-/myriadfs/home/ucacmqu/privacy_distill_test_runs}"
BASE_RUN_TAG="${BASE_RUN_TAG:-gpt-neo-1.3b-local}"
DATASET_TAG="${DATASET_TAG:-gpt-neo-fixed-20260419}"
RUN_TAG="${RUN_TAG:-gpt-neo-1.3b-local-placebo-npo-s13}"
SEED="${SEED:-13}"

LOCAL_DATASETS_DIR="$LOCAL_ROOT/data/datasets/$DATASET_TAG"
LOCAL_FORGET_DIR="$LOCAL_DATASETS_DIR/random_forget_train"
LOCAL_FORGET_META="$LOCAL_DATASETS_DIR/random_forget_meta.json"
LOCAL_PY="$LOCAL_ROOT/.venv/bin/python"

if [[ ! -d "$LOCAL_FORGET_DIR" ]]; then
  echo "Building local Neo repaired random-forget dataset: $LOCAL_FORGET_DIR"
  "$LOCAL_PY" "$LOCAL_ROOT/src/build_random_forget.py" \
    --dataset "bradfordlevy/BeanCounter" \
    --config "clean" \
    --split "train" \
    --cik-map "$LOCAL_ROOT/data/sec_index_10k.jsonl" \
    --tokenizer "EleutherAI/gpt-neo-1.3B" \
    --form-types "10-K" \
    --splits "$LOCAL_DATASETS_DIR/splits.json" \
    --stats-path "$LOCAL_ROOT/data/bean_counter_stats.jsonl" \
    --num-forget 50 \
    --min-tokens 200000 \
    --max-length 512 \
    --max-tokens-per-company 0 \
    --seed "$SEED" \
    --output "$LOCAL_FORGET_DIR"
fi

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
  '$REMOTE_ROOT/data/datasets/$DATASET_TAG' \
  '$REMOTE_ROOT/outputs/$RUN_TAG/reference_eval' \
  '$REMOTE_ROOT/outputs/$BASE_RUN_TAG/teachers' \
  '$REMOTE_ROOT/outputs/$BASE_RUN_TAG/students'"

echo "Syncing Neo placebo scripts"
"${SSH_RSYNC[@]}" \
  "$WORKSPACE_ROOT/cluster_repo/scripts/run_placebo_c6_npo_neo_1p3b.sh" \
  "$WORKSPACE_ROOT/cluster_repo/scripts/fetch_neo13_placebo_c6_npo.sh" \
  "$REMOTE_HOST:$REMOTE_ROOT/scripts/"

echo "Syncing repaired Neo random-forget dataset"
"${SSH_RSYNC[@]}" \
  "$LOCAL_FORGET_DIR" \
  "$LOCAL_FORGET_META" \
  "$REMOTE_HOST:$REMOTE_ROOT/data/datasets/$DATASET_TAG/"

echo "Syncing Neo reference target JSONs"
"${SSH_RSYNC[@]}" \
  "$LOCAL_ROOT/outputs/$BASE_RUN_TAG/mia/c1_student.json" \
  "$LOCAL_ROOT/outputs/$BASE_RUN_TAG/mia/c3_student.json" \
  "$REMOTE_HOST:$REMOTE_ROOT/outputs/$RUN_TAG/reference_eval/"

echo "Syncing required Neo model artifacts"
"${SSH_RSYNC[@]}" \
  "$LOCAL_ROOT/outputs/$BASE_RUN_TAG/teachers/c1" \
  "$REMOTE_HOST:$REMOTE_ROOT/outputs/$BASE_RUN_TAG/teachers/"
"${SSH_RSYNC[@]}" \
  "$LOCAL_ROOT/outputs/$BASE_RUN_TAG/students/c1" \
  "$LOCAL_ROOT/outputs/$BASE_RUN_TAG/students/c3" \
  "$REMOTE_HOST:$REMOTE_ROOT/outputs/$BASE_RUN_TAG/students/"

echo "Submitting Neo placebo job"
"${SSH_CMD[@]}" "REMOTE_ROOT='$REMOTE_ROOT' RUN_TAG='$RUN_TAG' BASE_RUN_TAG='$BASE_RUN_TAG' DATASET_TAG='$DATASET_TAG' bash -s" <<'REMOTE'
set -euo pipefail

ROOT="${REMOTE_ROOT:-/myriadfs/home/ucacmqu/privacy_distill_test_runs}"
RUN_TAG="${RUN_TAG:-gpt-neo-1.3b-local-placebo-npo-s13}"
BASE_RUN_TAG="${BASE_RUN_TAG:-gpt-neo-1.3b-local}"
DATASET_TAG="${DATASET_TAG:-gpt-neo-fixed-20260419}"
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

require_dir "$ROOT/data/datasets/$DATASET_TAG/distill"
require_dir "$ROOT/data/datasets/$DATASET_TAG/eval_target_holdout"
require_dir "$ROOT/data/datasets/$DATASET_TAG/eval_nonmember"
require_dir "$ROOT/data/datasets/eval_retain_holdout"
require_dir "$ROOT/data/datasets/$DATASET_TAG/random_forget_train"
require_dir "$ROOT/data/datasets/$DATASET_TAG/teacher_c2"
require_model_dir "$ROOT/outputs/$BASE_RUN_TAG/teachers/c1"
require_model_dir "$ROOT/outputs/$BASE_RUN_TAG/students/c1"
require_model_dir "$ROOT/outputs/$BASE_RUN_TAG/students/c3"
require_file "$ROOT/outputs/$RUN_TAG/reference_eval/c1_student.json"
require_file "$ROOT/outputs/$RUN_TAG/reference_eval/c3_student.json"
require_dir "$ROOT/scripts"
require_dir "$ROOT/cluster"

qsub -N neo13-placebo-c6 \
  -l h_rt=30:00:00 -l mem=8G -l gpu=4 \
  -pe smp 8 -ac allow=L \
  -v REPO_ROOT="$ROOT",RUN_TAG="$RUN_TAG",BASE_RUN_TAG="$BASE_RUN_TAG",DATASETS_DIR="$ROOT/data/datasets/$DATASET_TAG",RUN_CMD=./scripts/run_placebo_c6_npo_neo_1p3b.sh \
  cluster/qsub_run.sh
REMOTE

echo "Neo sync and submit complete."
