#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

RUN_TAG="${RUN_TAG:-pythia-1.4b}"
MODEL="${MODEL:-EleutherAI/pythia-1.4b}"
STUDENT="${STUDENT:-EleutherAI/pythia-410m}"
SEEDS="${SEEDS:-13 17 19}"
DATASETS_DIR="${DATASETS_DIR:-$ROOT/data/datasets/$RUN_TAG}"
SEED_ROOT="${SEED_ROOT:-$ROOT/outputs/$RUN_TAG/seed_reps}"
MAX_LENGTH="${MAX_LENGTH:-512}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-3}"
LR="${LR:-2e-5}"
DISTILL_LR="${DISTILL_LR:-$LR}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
OPTIM="${OPTIM:-adamw_8bit}"
BF16="${BF16:-1}"
UNLEARN_BATCH="${UNLEARN_BATCH:-2}"
UNLEARN_GRAD_ACCUM="${UNLEARN_GRAD_ACCUM:-16}"
DISTILL_BATCH="${DISTILL_BATCH:-2}"
DISTILL_GRAD_ACCUM="${DISTILL_GRAD_ACCUM:-16}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-2.0}"
DISTILL_ALPHA="${DISTILL_ALPHA:-0.5}"
DISTILL_MAX_GRAD_NORM="${DISTILL_MAX_GRAD_NORM:-1.0}"
ALPHA="${ALPHA:-1.0}"
BETA="${BETA:-1.0}"
CUDA_GPU="${CUDA_GPU:-0}"
DISTILL_CUDA_DEVICES="${DISTILL_CUDA_DEVICES:-0,1}"
DISTILL_TEACHER_DEVICE="${DISTILL_TEACHER_DEVICE:-cuda:1}"
DISTILL_TEACHER_DTYPE="${DISTILL_TEACHER_DTYPE:-float32}"
FORCE_DISTILL="${FORCE_DISTILL:-0}"
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

if [[ "$OPTIM" == "adamw_8bit" && "$BNB_AVAILABLE" != "1" ]]; then
  log "WARN: bitsandbytes not available; falling back from OPTIM=adamw_8bit to UNLEARN_OPTIM=adamw and DISTILL_OPTIM=$DISTILL_OPTIM_FALLBACK"
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
    log "FAIL seed_${seed}: missing seed-local c1 teacher at $c1_teacher"
    return 1
  fi

  if ! model_ready "$c5_teacher"; then
    run_logged "seed${seed}_unlearn_c5" \
      "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$CUDA_GPU $PY $ROOT/src/unlearn_teacher.py --model '$c1_teacher' --forget-dataset '$forget_data' --retain-dataset '$retain_data' --output '$c5_teacher' $(bf16_flag) --optim '$UNLEARN_OPTIM' --batch-size $UNLEARN_BATCH --grad-accum $UNLEARN_GRAD_ACCUM --epochs 1 --lr $LR --alpha $ALPHA --beta $BETA --seed $seed"
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
    run_logged "seed${seed}_distill_c5" \
      "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$DISTILL_CUDA_DEVICES $PY $ROOT/src/distill_student.py --teacher '$c5_teacher' --student '$STUDENT' --teacher-device '$DISTILL_TEACHER_DEVICE' --teacher-dtype '$DISTILL_TEACHER_DTYPE' --dataset '$distill_data' --output '$c5_student' --max-length $MAX_LENGTH --epochs $DISTILL_EPOCHS --lr $DISTILL_LR --warmup-steps $WARMUP_STEPS --per-device-batch $DISTILL_BATCH --grad-accum $DISTILL_GRAD_ACCUM --optim '$DISTILL_OPTIM' --temperature $DISTILL_TEMPERATURE --alpha $DISTILL_ALPHA --max-grad-norm $DISTILL_MAX_GRAD_NORM --seed $seed --save-steps 999999 --logging-steps 50 $(bf16_flag)"
  else
    log "SKIP seed${seed}_distill_c5 (model exists)"
  fi

  if [[ ! -f "$mia_dir/c5_student.json" ]]; then
    run_logged "seed${seed}_eval_mia_c5_student" \
      "CUDA_VISIBLE_DEVICES=$CUDA_GPU $PY $ROOT/src/eval_mia.py --model '$c5_student' --target-holdout '$holdout_data' --nonmember '$nonmember_data' --holdout-map '$holdout_map' --output '$mia_dir/c5_student.json' --batch-size 4 --max-length $MAX_LENGTH $(bf16_flag)"
  else
    log "SKIP seed${seed}_eval_mia_c5_student (output exists)"
  fi

  if [[ ! -f "$mia_dir/c5_teacher.json" ]]; then
    run_logged "seed${seed}_eval_mia_c5_teacher" \
      "CUDA_VISIBLE_DEVICES=$CUDA_GPU $PY $ROOT/src/eval_mia.py --model '$c5_teacher' --target-holdout '$holdout_data' --nonmember '$nonmember_data' --holdout-map '$holdout_map' --output '$mia_dir/c5_teacher.json' --batch-size 4 --max-length $MAX_LENGTH $(bf16_flag)"
  else
    log "SKIP seed${seed}_eval_mia_c5_teacher (output exists)"
  fi

  if [[ ! -f "$mia_dir/utility_c5_student.json" ]]; then
    run_logged "seed${seed}_utility_c5_student" \
      "CUDA_VISIBLE_DEVICES=$CUDA_GPU $PY $ROOT/src/eval_ppl.py --model '$c5_student' --dataset '$distill_data' --output '$mia_dir/utility_c5_student.json' --batch-size 4 --max-samples 500 --seed $seed $(bf16_flag)"
  else
    log "SKIP seed${seed}_utility_c5_student (output exists)"
  fi

  if [[ ! -f "$mia_dir/utility_c5_teacher.json" ]]; then
    run_logged "seed${seed}_utility_c5_teacher" \
      "CUDA_VISIBLE_DEVICES=$CUDA_GPU $PY $ROOT/src/eval_ppl.py --model '$c5_teacher' --dataset '$distill_data' --output '$mia_dir/utility_c5_teacher.json' --batch-size 4 --max-samples 500 --seed $seed $(bf16_flag)"
  else
    log "SKIP seed${seed}_utility_c5_teacher (output exists)"
  fi

  if [[ ! -f "$mia_dir/utility_c5_student_holdout.json" ]]; then
    run_logged "seed${seed}_holdout_c5_student" \
      "CUDA_VISIBLE_DEVICES=$CUDA_GPU $PY $ROOT/src/eval_ppl.py --model '$c5_student' --dataset '$holdout_utility_data' --output '$mia_dir/utility_c5_student_holdout.json' --batch-size 4 --max-samples 250 --seed $seed $(bf16_flag)"
  else
    log "SKIP seed${seed}_holdout_c5_student (output exists)"
  fi

  if [[ ! -f "$mia_dir/utility_c5_teacher_holdout.json" ]]; then
    run_logged "seed${seed}_holdout_c5_teacher" \
      "CUDA_VISIBLE_DEVICES=$CUDA_GPU $PY $ROOT/src/eval_ppl.py --model '$c5_teacher' --dataset '$holdout_utility_data' --output '$mia_dir/utility_c5_teacher_holdout.json' --batch-size 4 --max-samples 250 --seed $seed $(bf16_flag)"
  else
    log "SKIP seed${seed}_holdout_c5_teacher (output exists)"
  fi
}

log "Pythia 1.4B c5 seed run start: seeds=$SEEDS run_tag=$RUN_TAG cuda_gpu=$CUDA_GPU distill_cuda_devices=$DISTILL_CUDA_DEVICES distill_teacher_device=$DISTILL_TEACHER_DEVICE force_distill=$FORCE_DISTILL"

for seed in $SEEDS; do
  run_seed_chain "$seed"
done

log "Pythia 1.4B c5 seed run complete."
