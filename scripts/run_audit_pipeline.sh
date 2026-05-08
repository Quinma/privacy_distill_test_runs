#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG_PATH="${1:-${AUDIT_CONFIG:-$ROOT/configs/audit.env.example}}"

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: missing config file: $CONFIG_PATH" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONFIG_PATH"

export MODEL="${MODEL:-EleutherAI/pythia-1.4b}"
export STUDENT="${STUDENT:-EleutherAI/pythia-410m}"
export RUN_TAG="${RUN_TAG:-audit-run}"
export TOKENIZER="${TOKENIZER:-$MODEL}"
export DATASET="${DATASET:-bradfordlevy/BeanCounter}"
export CONFIG="${CONFIG:-clean}"
export REVISION="${REVISION:-}"
export SPLIT="${SPLIT:-train}"
export DATA_FILES="${DATA_FILES:-}"
export CIK_MAP="${CIK_MAP:-$ROOT/data/sec_index_10k.jsonl}"
export FORM_TYPES="${FORM_TYPES:-10-K}"
export TEXT_FIELD="${TEXT_FIELD:-text}"
export CIK_FIELD="${CIK_FIELD:-cik}"
export FORM_FIELD="${FORM_FIELD:-type_filing}"
export DATE_FIELD="${DATE_FIELD:-date}"
export STATS_PATH="${STATS_PATH:-$ROOT/data/bean_counter_stats.jsonl}"
export TOP_N="${TOP_N:-100}"
export N_TARGETS="${N_TARGETS:-50}"
export HOLDOUT_MAX="${HOLDOUT_MAX:-5}"
export HOLDOUT_FRAC="${HOLDOUT_FRAC:-0.2}"
export BACKGROUND_TOKENS="${BACKGROUND_TOKENS:-50000000}"
export NONMEMBER_TOKENS="${NONMEMBER_TOKENS:-15000000}"
export MAX_TOKENS_PER_COMPANY="${MAX_TOKENS_PER_COMPANY:-500000}"
export MIN_TOKENS="${MIN_TOKENS:-200000}"
export LOG_EVERY="${LOG_EVERY:-5000}"

export DATASETS_DIR="${DATASETS_DIR:-$ROOT/data/datasets/$RUN_TAG}"
export TEACHERS_DIR="${TEACHERS_DIR:-$ROOT/outputs/$RUN_TAG/teachers}"
export STUDENTS_DIR="${STUDENTS_DIR:-$ROOT/outputs/$RUN_TAG/students}"
export MIA_DIR="${MIA_DIR:-$ROOT/outputs/$RUN_TAG/mia}"

export MAX_LENGTH="${MAX_LENGTH:-512}"
export BF16="${BF16:-1}"
export TRAIN_BATCH="${TRAIN_BATCH:-2}"
export TRAIN_ACCUM="${TRAIN_ACCUM:-16}"
export TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
export TRAIN_LR="${TRAIN_LR:-2e-5}"
export WARMUP_STEPS="${WARMUP_STEPS:-500}"
export OPTIM="${OPTIM:-adamw_torch}"
export WORKERS="${WORKERS:-4}"
export VISIBLE_GPUS="${VISIBLE_GPUS:-0,1,2}"
export TRAIN_GPU_C1="${TRAIN_GPU_C1:-0}"
export TRAIN_GPU_C2="${TRAIN_GPU_C2:-1}"
export TRAIN_GPU_C3="${TRAIN_GPU_C3:-2}"
export DISTILL_GPU_C1="${DISTILL_GPU_C1:-0}"
export DISTILL_GPU_C2="${DISTILL_GPU_C2:-1}"
export DISTILL_GPU_C3="${DISTILL_GPU_C3:-2}"
export UNLEARN_GPU="${UNLEARN_GPU:-0}"
export EVAL_GPU="${EVAL_GPU:-0}"
export UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-3}"
export UNLEARN_LR="${UNLEARN_LR:-2e-5}"
export RETAIN_WEIGHT="${RETAIN_WEIGHT:-1.0}"
export NPO_BETA="${NPO_BETA:-0.1}"
export DISTILL_TEACHER_DEVICE="${DISTILL_TEACHER_DEVICE:-cuda}"
export DISTILL_TEACHER_DTYPE="${DISTILL_TEACHER_DTYPE:-auto}"

export RETAIN_HOLDOUT_DATASET="${RETAIN_HOLDOUT_DATASET:-$ROOT/data/datasets/eval_retain_holdout}"
export TARGET_HOLDOUT_DATASET="${TARGET_HOLDOUT_DATASET:-$DATASETS_DIR/eval_target_holdout}"
export NONMEMBER_DATASET="${NONMEMBER_DATASET:-$DATASETS_DIR/eval_nonmember}"
export HOLDOUT_MAP="${HOLDOUT_MAP:-$DATASETS_DIR/holdout_map.json}"
export SEED="${SEED:-13}"

run_stage() {
  local label="$1"
  shift
  echo "[run_audit_pipeline] START $label"
  "$@"
  echo "[run_audit_pipeline] DONE $label"
}

[[ "${RUN_GATE:-0}" == "1" ]] && run_stage gate bash "$ROOT/scripts/run_gate.sh"
[[ "${RUN_BUILD:-0}" == "1" ]] && run_stage build bash "$ROOT/scripts/run_build.sh"
[[ "${RUN_TRAIN:-1}" == "1" ]] && run_stage train_teachers bash "$ROOT/scripts/run_train_teachers.sh"
[[ "${RUN_DISTILL:-1}" == "1" ]] && run_stage distill_baselines bash "$ROOT/scripts/run_distill.sh"
[[ "${RUN_C6_NPO:-1}" == "1" ]] && run_stage c6_npo bash "$ROOT/scripts/run_c6_npo.sh"
[[ "${RUN_STANDARD_EVAL:-1}" == "1" ]] && run_stage standard_eval bash "$ROOT/scripts/run_eval.sh"
[[ "${RUN_RETAIN_AUDIT:-1}" == "1" ]] && run_stage retain_audit bash "$ROOT/scripts/run_retain_audit.sh"
