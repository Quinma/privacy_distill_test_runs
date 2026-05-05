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

UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-3}"
UNLEARN_LR="${UNLEARN_LR:-2e-5}"
UNLEARN_BATCH="${UNLEARN_BATCH:-2}"
UNLEARN_GRAD_ACCUM="${UNLEARN_GRAD_ACCUM:-16}"
ALPHA="${ALPHA:-1.0}"
BETA="${BETA:-1.0}"

DISTILL_EPOCHS="${DISTILL_EPOCHS:-3}"
DISTILL_LR="${DISTILL_LR:-2e-5}"
DISTILL_WARMUP_STEPS="${DISTILL_WARMUP_STEPS:-500}"
DISTILL_BATCH="${DISTILL_BATCH:-2}"
DISTILL_GRAD_ACCUM="${DISTILL_GRAD_ACCUM:-16}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.5}"
DISTILL_MAX_GRAD_NORM="${DISTILL_MAX_GRAD_NORM:-1.0}"
DISTILL_SAVE_STEPS="${DISTILL_SAVE_STEPS:-2000}"

OPTIM="${OPTIM:-adamw_8bit}"
CUDA_GPU="${CUDA_GPU:-0}"
TRAIN_GPU="${TRAIN_GPU:-$CUDA_GPU}"
UNLEARN_GPU="${UNLEARN_GPU:-$CUDA_GPU}"
EVAL_GPU="${EVAL_GPU:-$CUDA_GPU}"
DISTILL_CUDA_DEVICES="${DISTILL_CUDA_DEVICES:-0,1}"
DISTILL_TEACHER_DEVICE="${DISTILL_TEACHER_DEVICE:-cuda:1}"
DISTILL_TEACHER_DTYPE="${DISTILL_TEACHER_DTYPE:-float32}"
FORCE_DISTILL="${FORCE_DISTILL:-0}"
AUTO_TRAIN_C1="${AUTO_TRAIN_C1:-1}"

LOG_DIR="${LOG_DIR:-$ROOT/outputs/logs/${RUN_TAG}_seed_c5_$(date +%Y%m%d_%H%M%S)}"
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

holdout_utility_dataset() {
  if [[ -d "$DATASETS_DIR/eval_target_holdout_tok" ]]; then
    printf '%s\n' "$DATASETS_DIR/eval_target_holdout_tok"
  else
    printf '%s\n' "$DATASETS_DIR/eval_target_holdout"
  fi
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

BNB_AVAILABLE=0
if "$PY" -c "import bitsandbytes" >/dev/null 2>&1; then
  BNB_AVAILABLE=1
fi

TRAIN_OPTIM="$OPTIM"
UNLEARN_OPTIM="$OPTIM"
DISTILL_OPTIM="$OPTIM"

DISTILL_OPTIM_FALLBACK="$("$PY" - <<'PY'
try:
    from transformers.training_args import OptimizerNames
except Exception:
    print("adamw_torch")
    raise SystemExit(0)
names = {item.value for item in OptimizerNames}
for candidate in ("adamw_torch", "adamw"):
    if candidate in names:
        print(candidate)
        break
else:
    print("adamw_torch")
PY
)"

TRAIN_OPTIM_FALLBACK="$DISTILL_OPTIM_FALLBACK"

if [[ "$OPTIM" == "adamw_8bit" && "$BNB_AVAILABLE" != "1" ]]; then
  log "WARN: bitsandbytes not available; falling back from OPTIM=adamw_8bit to TRAIN_OPTIM=$TRAIN_OPTIM_FALLBACK UNLEARN_OPTIM=adamw DISTILL_OPTIM=$DISTILL_OPTIM_FALLBACK"
  TRAIN_OPTIM="$TRAIN_OPTIM_FALLBACK"
  UNLEARN_OPTIM="adamw"
  DISTILL_OPTIM="$DISTILL_OPTIM_FALLBACK"
fi

run_seed_chain() {
  local seed="$1"
  local out_root="$SEED_ROOT/seed_${seed}"
  local teachers_dir="$out_root/teachers"
  local students_dir="$out_root/students"
  local mia_dir="$out_root/mia"
  local c1_teacher="$teachers_dir/c1"
  local c5_teacher="$teachers_dir/c5_unlearn"
  local c5_student="$students_dir/c5"
  local teacher_c1_data="$DATASETS_DIR/teacher_c1"
  local distill_data="$DATASETS_DIR/distill"
  local holdout_data="$DATASETS_DIR/eval_target_holdout"
  local holdout_utility_data
  local nonmember_data="$DATASETS_DIR/eval_nonmember"
  local holdout_map="$DATASETS_DIR/holdout_map.json"
  local forget_data="$DATASETS_DIR/target_train"
  local retain_data="$DATASETS_DIR/teacher_c2"

  mkdir -p "$teachers_dir" "$students_dir" "$mia_dir"
  holdout_utility_data="$(holdout_utility_dataset)"

  if ! model_ready "$c1_teacher"; then
    if [[ "$AUTO_TRAIN_C1" != "1" ]]; then
      log "FAIL seed_${seed}: missing seed-local c1 teacher at $c1_teacher and AUTO_TRAIN_C1=0"
      return 1
    fi

    local total_c1 resume_c1 model_c1 resume_arg_c1 max_steps_arg_c1 warmup_arg_c1 step_c1 remain_c1
    total_c1="$(total_steps "$teacher_c1_data" "$TRAIN_EPOCHS" "$TRAIN_BATCH" "$TRAIN_GRAD_ACCUM")"
    resume_c1="$(latest_ckpt "$c1_teacher")"
    model_c1="$MODEL"
    resume_arg_c1=""
    max_steps_arg_c1=""
    warmup_arg_c1="--warmup-steps $TRAIN_WARMUP_STEPS"

    if [[ -n "$resume_c1" ]]; then
      step_c1="$(ckpt_step "$resume_c1")"
      remain_c1=$((total_c1 - step_c1))
      if (( remain_c1 <= 0 )); then
        log "SKIP seed${seed}_train_c1 (checkpoint already complete at $resume_c1)"
      else
        pick_model_resume "$MODEL" "$resume_c1" model_c1 resume_arg_c1
        if [[ -z "$resume_arg_c1" ]]; then
          max_steps_arg_c1="--max-steps $remain_c1"
          warmup_arg_c1="--warmup-steps 0"
        fi
        run_logged "seed${seed}_train_c1" \
          "CUDA_VISIBLE_DEVICES=$TRAIN_GPU $PY $ROOT/src/train_teacher.py --model '$model_c1' --dataset '$teacher_c1_data' --output '$c1_teacher' --max-length $MAX_LENGTH --epochs $TRAIN_EPOCHS --lr $TRAIN_LR $warmup_arg_c1 $max_steps_arg_c1 --per-device-batch $TRAIN_BATCH --grad-accum $TRAIN_GRAD_ACCUM --optim '$TRAIN_OPTIM' --seed $seed $resume_arg_c1 $(bf16_flag)"
      fi
    else
      run_logged "seed${seed}_train_c1" \
        "CUDA_VISIBLE_DEVICES=$TRAIN_GPU $PY $ROOT/src/train_teacher.py --model '$MODEL' --dataset '$teacher_c1_data' --output '$c1_teacher' --max-length $MAX_LENGTH --epochs $TRAIN_EPOCHS --lr $TRAIN_LR --warmup-steps $TRAIN_WARMUP_STEPS --per-device-batch $TRAIN_BATCH --grad-accum $TRAIN_GRAD_ACCUM --optim '$TRAIN_OPTIM' --seed $seed $(bf16_flag)"
    fi
  else
    log "SKIP seed${seed}_train_c1 (model exists)"
  fi

  if ! model_ready "$c5_teacher"; then
    run_logged "seed${seed}_unlearn_c5" \
      "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$UNLEARN_GPU $PY $ROOT/src/unlearn_teacher.py --model '$c1_teacher' --forget-dataset '$forget_data' --retain-dataset '$retain_data' --output '$c5_teacher' $(bf16_flag) --optim '$UNLEARN_OPTIM' --batch-size $UNLEARN_BATCH --grad-accum $UNLEARN_GRAD_ACCUM --epochs $UNLEARN_EPOCHS --lr $UNLEARN_LR --alpha $ALPHA --beta $BETA --seed $seed"
  else
    log "SKIP seed${seed}_unlearn_c5 (model exists)"
  fi

  if [[ "$FORCE_DISTILL" == "1" ]]; then
    log "FORCE seed${seed}_distill_c5 (removing prior student and derived eval outputs)"
    rm -rf "$c5_student"
    rm -f \
      "$mia_dir/c5_student.json" \
      "$mia_dir/utility_c5_student.json" \
      "$mia_dir/utility_c5_student_holdout.json"
  fi

  if ! model_ready "$c5_student"; then
    local total_distill resume_distill student_model resume_arg_distill max_steps_arg_distill warmup_arg_distill step_distill remain_distill
    total_distill="$(total_steps "$distill_data" "$DISTILL_EPOCHS" "$DISTILL_BATCH" "$DISTILL_GRAD_ACCUM")"
    resume_distill="$(latest_ckpt "$c5_student")"
    student_model="$STUDENT"
    resume_arg_distill=""
    max_steps_arg_distill=""
    warmup_arg_distill="--warmup-steps $DISTILL_WARMUP_STEPS"

    if [[ -n "$resume_distill" ]]; then
      step_distill="$(ckpt_step "$resume_distill")"
      remain_distill=$((total_distill - step_distill))
      if (( remain_distill <= 0 )); then
        log "SKIP seed${seed}_distill_c5 (checkpoint already complete at $resume_distill)"
      else
        pick_model_resume "$STUDENT" "$resume_distill" student_model resume_arg_distill
        if [[ -z "$resume_arg_distill" ]]; then
          max_steps_arg_distill="--max-steps $remain_distill"
          warmup_arg_distill="--warmup-steps 0"
        fi
        run_logged "seed${seed}_distill_c5" \
          "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$DISTILL_CUDA_DEVICES $PY $ROOT/src/distill_student.py --teacher '$c5_teacher' --student '$student_model' --teacher-device '$DISTILL_TEACHER_DEVICE' --teacher-dtype '$DISTILL_TEACHER_DTYPE' --dataset '$distill_data' --output '$c5_student' --max-length $MAX_LENGTH --epochs $DISTILL_EPOCHS --lr $DISTILL_LR $warmup_arg_distill $max_steps_arg_distill --per-device-batch $DISTILL_BATCH --grad-accum $DISTILL_GRAD_ACCUM --optim '$DISTILL_OPTIM' --temperature $DISTILL_TEMPERATURE --alpha $DISTILL_ALPHA --max-grad-norm $DISTILL_MAX_GRAD_NORM --save-steps $DISTILL_SAVE_STEPS --logging-steps 50 --seed $seed $resume_arg_distill $(bf16_flag)"
      fi
    else
      run_logged "seed${seed}_distill_c5" \
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$DISTILL_CUDA_DEVICES $PY $ROOT/src/distill_student.py --teacher '$c5_teacher' --student '$STUDENT' --teacher-device '$DISTILL_TEACHER_DEVICE' --teacher-dtype '$DISTILL_TEACHER_DTYPE' --dataset '$distill_data' --output '$c5_student' --max-length $MAX_LENGTH --epochs $DISTILL_EPOCHS --lr $DISTILL_LR --warmup-steps $DISTILL_WARMUP_STEPS --per-device-batch $DISTILL_BATCH --grad-accum $DISTILL_GRAD_ACCUM --optim '$DISTILL_OPTIM' --temperature $DISTILL_TEMPERATURE --alpha $DISTILL_ALPHA --max-grad-norm $DISTILL_MAX_GRAD_NORM --save-steps $DISTILL_SAVE_STEPS --logging-steps 50 --seed $seed $(bf16_flag)"
    fi
  else
    log "SKIP seed${seed}_distill_c5 (model exists)"
  fi

  if [[ ! -f "$mia_dir/c5_student.json" ]]; then
    run_logged "seed${seed}_eval_mia_c5_student" \
      "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY $ROOT/src/eval_mia.py --model '$c5_student' --target-holdout '$holdout_data' --nonmember '$nonmember_data' --holdout-map '$holdout_map' --output '$mia_dir/c5_student.json' --batch-size 4 --max-length $MAX_LENGTH $(bf16_flag)"
  else
    log "SKIP seed${seed}_eval_mia_c5_student (output exists)"
  fi

  if [[ ! -f "$mia_dir/c5_teacher.json" ]]; then
    run_logged "seed${seed}_eval_mia_c5_teacher" \
      "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY $ROOT/src/eval_mia.py --model '$c5_teacher' --target-holdout '$holdout_data' --nonmember '$nonmember_data' --holdout-map '$holdout_map' --output '$mia_dir/c5_teacher.json' --batch-size 4 --max-length $MAX_LENGTH $(bf16_flag)"
  else
    log "SKIP seed${seed}_eval_mia_c5_teacher (output exists)"
  fi

  if [[ ! -f "$mia_dir/utility_c5_student.json" ]]; then
    run_logged "seed${seed}_utility_c5_student" \
      "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY $ROOT/src/eval_ppl.py --model '$c5_student' --dataset '$distill_data' --output '$mia_dir/utility_c5_student.json' --batch-size 4 --max-samples 500 --seed $seed $(bf16_flag)"
  else
    log "SKIP seed${seed}_utility_c5_student (output exists)"
  fi

  if [[ ! -f "$mia_dir/utility_c5_teacher.json" ]]; then
    run_logged "seed${seed}_utility_c5_teacher" \
      "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY $ROOT/src/eval_ppl.py --model '$c5_teacher' --dataset '$distill_data' --output '$mia_dir/utility_c5_teacher.json' --batch-size 4 --max-samples 500 --seed $seed $(bf16_flag)"
  else
    log "SKIP seed${seed}_utility_c5_teacher (output exists)"
  fi

  if [[ ! -f "$mia_dir/utility_c5_student_holdout.json" ]]; then
    run_logged "seed${seed}_holdout_c5_student" \
      "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY $ROOT/src/eval_ppl.py --model '$c5_student' --dataset '$holdout_utility_data' --output '$mia_dir/utility_c5_student_holdout.json' --batch-size 4 --max-samples 250 --seed $seed $(bf16_flag)"
  else
    log "SKIP seed${seed}_holdout_c5_student (output exists)"
  fi

  if [[ ! -f "$mia_dir/utility_c5_teacher_holdout.json" ]]; then
    run_logged "seed${seed}_holdout_c5_teacher" \
      "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY $ROOT/src/eval_ppl.py --model '$c5_teacher' --dataset '$holdout_utility_data' --output '$mia_dir/utility_c5_teacher_holdout.json' --batch-size 4 --max-samples 250 --seed $seed $(bf16_flag)"
  else
    log "SKIP seed${seed}_holdout_c5_teacher (output exists)"
  fi
}

log "Neo 1.3B c5 seed run start: seeds=$SEEDS run_tag=$RUN_TAG train_gpu=$TRAIN_GPU unlearn_gpu=$UNLEARN_GPU eval_gpu=$EVAL_GPU distill_cuda_devices=$DISTILL_CUDA_DEVICES distill_teacher_device=$DISTILL_TEACHER_DEVICE force_distill=$FORCE_DISTILL auto_train_c1=$AUTO_TRAIN_C1"

for seed in $SEEDS; do
  run_seed_chain "$seed"
done

log "Neo 1.3B c5 seed run complete."
