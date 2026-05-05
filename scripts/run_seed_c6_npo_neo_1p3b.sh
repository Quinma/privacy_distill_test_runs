#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

RUN_TAG="${RUN_TAG:-gpt-neo-1.3b-local}"
MODEL="${MODEL:-EleutherAI/gpt-neo-1.3B}"
STUDENT="${STUDENT:-EleutherAI/gpt-neo-125M}"
SEEDS="${SEEDS:-13 17 19}"
DATASETS_DIR="${DATASETS_DIR:-$ROOT/data/datasets/$RUN_TAG}"
SEED_ROOT="${SEED_ROOT:-$ROOT/outputs/$RUN_TAG/seed_reps}"

MAX_LENGTH="${MAX_LENGTH:-512}"
BF16="${BF16:-1}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-3}"
TRAIN_LR="${TRAIN_LR:-2e-5}"
TRAIN_WARMUP_STEPS="${TRAIN_WARMUP_STEPS:-500}"
TRAIN_BATCH="${TRAIN_BATCH:-2}"
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-16}"
TRAIN_OPTIM="${TRAIN_OPTIM:-adamw_torch}"

VISIBLE_GPUS="${VISIBLE_GPUS:-0,1,2}"
UNLEARN_NPROC="${UNLEARN_NPROC:-3}"
DISTILL_NPROC="${DISTILL_NPROC:-3}"
EVAL_GPU="${EVAL_GPU:-3}"
UNLEARN_CPU_OFFLOAD="${UNLEARN_CPU_OFFLOAD:-0}"

AUTO_TRAIN_C1="${AUTO_TRAIN_C1:-1}"

LOG_DIR="${LOG_DIR:-$ROOT/outputs/logs/${RUN_TAG}_seed_c6_$(date +%Y%m%d_%H%M%S)}"
RUN_LOG="$LOG_DIR/run.log"
mkdir -p "$LOG_DIR"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$RUN_LOG"; }

bf16_flag() {
  if [[ "$BF16" == "1" ]]; then
    printf '%s' "--bf16"
  fi
}

model_ready() {
  local dir="$1"
  [[ -f "$dir/model.safetensors" || -f "$dir/pytorch_model.bin" || -f "$dir/model.safetensors.index.json" || -f "$dir/model-00001-of-00002.safetensors" ]]
}

latest_ckpt() {
  local dir="$1"
  ls -dt "$dir"/checkpoint-* 2>/dev/null | head -n 1 || true
}

ckpt_step() {
  local ckpt="$1"
  if [[ -z "$ckpt" ]]; then
    echo ""
    return
  fi
  basename "$ckpt" | sed 's/^checkpoint-//'
}

total_steps() {
  local dataset="$1"
  local epochs="$2"
  local per_device_batch="$3"
  local grad_accum="$4"
  "$PY" - <<PY
import math
import datasets
ds = datasets.load_from_disk("$dataset")
steps_per_epoch = math.ceil(len(ds) / ($per_device_batch * $grad_accum))
print(int(steps_per_epoch * $epochs))
PY
}

pick_model_resume() {
  local base_model="$1"
  local ckpt_dir="$2"
  local model_out="$3"
  local resume_out="$4"

  local chosen_model="$base_model"
  local resume_arg=""
  if [[ -n "$ckpt_dir" && -d "$ckpt_dir" ]]; then
    if [[ -f "$ckpt_dir/optimizer.pt" ]]; then
      resume_arg="--resume $ckpt_dir"
    else
      chosen_model="$ckpt_dir"
    fi
  fi
  printf -v "$model_out" "%s" "$chosen_model"
  printf -v "$resume_out" "%s" "$resume_arg"
}

run_logged() {
  local name="$1"
  local log_file="$LOG_DIR/${name}.log"
  shift
  log "START $name"
  echo "[$(ts)] CMD: $*" | tee -a "$log_file"
  set +e
  bash -lc "$*" 2>&1 | tee -a "$log_file"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ $status -ne 0 ]]; then
    log "FAIL $name (exit $status)"
    return $status
  fi
  log "DONE $name"
}

run_seed() {
  local seed="$1"
  local out_root="$SEED_ROOT/seed_${seed}"
  local teachers_dir="$out_root/teachers"
  local students_dir="$out_root/students"
  local mia_dir="$out_root/mia"
  local c1_teacher="$teachers_dir/c1"
  local c1_data="$DATASETS_DIR/teacher_c1"
  local baseline_cache="$mia_dir/utility_c1_teacher.json"
  local seed_log_dir="$LOG_DIR/seed_${seed}"

  mkdir -p "$teachers_dir" "$students_dir" "$mia_dir" "$seed_log_dir"

  if ! model_ready "$c1_teacher"; then
    if [[ "$AUTO_TRAIN_C1" != "1" ]]; then
      log "FAIL seed_${seed}: missing seed-local c1 teacher at $c1_teacher and AUTO_TRAIN_C1=0"
      return 1
    fi

    local total_c1 resume_c1 model_c1 resume_arg_c1 max_steps_arg_c1 warmup_arg_c1 step_c1 remain_c1
    total_c1="$(total_steps "$c1_data" "$TRAIN_EPOCHS" "$TRAIN_BATCH" "$TRAIN_GRAD_ACCUM")"
    resume_c1="$(latest_ckpt "$c1_teacher")"
    model_c1="$MODEL"
    resume_arg_c1=""
    max_steps_arg_c1=""
    warmup_arg_c1="--warmup-steps $TRAIN_WARMUP_STEPS"

    if [[ -n "$resume_c1" ]]; then
      step_c1="$(ckpt_step "$resume_c1")"
      remain_c1=$((total_c1 - step_c1))
      if (( remain_c1 <= 0 )); then
        log "SKIP seed_${seed}_train_c1 (checkpoint already complete at $resume_c1)"
      else
        pick_model_resume "$MODEL" "$resume_c1" model_c1 resume_arg_c1
        if [[ -z "$resume_arg_c1" ]]; then
          max_steps_arg_c1="--max-steps $remain_c1"
          warmup_arg_c1="--warmup-steps 0"
        fi
        run_logged "seed_${seed}_train_c1" \
          "CUDA_VISIBLE_DEVICES=0 $PY $ROOT/src/train_teacher.py --model '$model_c1' --dataset '$c1_data' --output '$c1_teacher' --max-length $MAX_LENGTH --epochs $TRAIN_EPOCHS --lr $TRAIN_LR $warmup_arg_c1 $max_steps_arg_c1 --per-device-batch $TRAIN_BATCH --grad-accum $TRAIN_GRAD_ACCUM --optim '$TRAIN_OPTIM' --seed $seed $resume_arg_c1 $(bf16_flag)"
      fi
    else
      run_logged "seed_${seed}_train_c1" \
        "CUDA_VISIBLE_DEVICES=0 $PY $ROOT/src/train_teacher.py --model '$MODEL' --dataset '$c1_data' --output '$c1_teacher' --max-length $MAX_LENGTH --epochs $TRAIN_EPOCHS --lr $TRAIN_LR --warmup-steps $TRAIN_WARMUP_STEPS --per-device-batch $TRAIN_BATCH --grad-accum $TRAIN_GRAD_ACCUM --optim '$TRAIN_OPTIM' --seed $seed $(bf16_flag)"
    fi
  else
    log "SKIP seed_${seed}_train_c1 (model exists)"
  fi

  log "START seed_${seed}_c6"
  (
    export MODEL="$MODEL"
    export STUDENT="$STUDENT"
    export RUN_TAG="$RUN_TAG"
    export DATASETS_DIR="$DATASETS_DIR"
    export OUT_ROOT="$out_root"
    export TEACHERS_DIR="$teachers_dir"
    export STUDENTS_DIR="$students_dir"
    export MIA_DIR="$mia_dir"
    export POLICY_INIT="$c1_teacher"
    export REF_MODEL="$c1_teacher"
    export UNLEARN_OUT="$teachers_dir/c6_unlearn"
    export STUDENT_OUT="$students_dir/c6"
    export SEED="$seed"
    export BASELINE_C1_UTILITY_JSON="$baseline_cache"
    export BASELINE_C1_UTILITY_CACHE="$baseline_cache"
    export VISIBLE_GPUS="$VISIBLE_GPUS"
    export UNLEARN_NPROC="$UNLEARN_NPROC"
    export DISTILL_NPROC="$DISTILL_NPROC"
    export EVAL_GPU="$EVAL_GPU"
    export UNLEARN_CPU_OFFLOAD="$UNLEARN_CPU_OFFLOAD"
    export LOG_DIR="$seed_log_dir"
    bash "$ROOT/scripts/run_c6_npo_neo_1p3b.sh"
  )
  log "DONE seed_${seed}_c6"
}

log "Neo 1.3B seeded C6 start: seeds=$SEEDS visible_gpus=$VISIBLE_GPUS eval_gpu=$EVAL_GPU cpu_offload=$UNLEARN_CPU_OFFLOAD auto_train_c1=$AUTO_TRAIN_C1"

for seed in $SEEDS; do
  run_seed "$seed"
done

log "Neo 1.3B seeded C6 complete."
