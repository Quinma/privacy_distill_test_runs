#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

SEEDS="${SEEDS:-13 17 19}"
MODEL="${MODEL:-EleutherAI/pythia-1.4b}"
STUDENT="${STUDENT:-EleutherAI/pythia-410m}"

DATASETS_DIR="${DATASETS_DIR:-$ROOT/data/datasets/pythia-1.4b}"
TEACHER_C1_DATA="$DATASETS_DIR/teacher_c1"
TEACHER_C3_DATA="$DATASETS_DIR/teacher_c3"
RETAIN_DATA="$DATASETS_DIR/teacher_c2"
FORGET_DATA="$DATASETS_DIR/target_train"

MAX_LENGTH="${MAX_LENGTH:-512}"
# Match existing 1.4B runs (3 epochs) unless overridden.
EPOCHS="${EPOCHS:-3}"
LR="${LR:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
OPTIM="${OPTIM:-adamw_8bit}"
BF16_FLAG="--bf16"

# Unlearning hyperparams (C5m)
C5_ALPHA="${C5_ALPHA:-0.1}"
C5_BETA="${C5_BETA:-1.0}"
C5_KL_MODEL="${C5_KL_MODEL:-$MODEL}"
C5_KL_WEIGHT="${C5_KL_WEIGHT:-0.2}"
C5_KL_DEVICE="${C5_KL_DEVICE:-cuda}"
C5_KL_EVERY="${C5_KL_EVERY:-10}"
C5_EARLY_PATIENCE="${C5_EARLY_PATIENCE:-10}"
C5_EARLY_MIN_STEPS="${C5_EARLY_MIN_STEPS:--1}"
C5_EARLY_EVAL_EVERY="${C5_EARLY_EVAL_EVERY:-50}"

# GPU settings
TRAIN_GPU="${TRAIN_GPU:-0}"
DISTILL_GPU="${DISTILL_GPU:-0}"
UNLEARN_GPU="${UNLEARN_GPU:-0}"

echo "[seed-reps] Seeds: $SEEDS"
echo "[seed-reps] Using DATASETS_DIR=$DATASETS_DIR"

for SEED in $SEEDS; do
  echo "[seed-reps] ===== Seed $SEED ====="
  OUT_ROOT="$ROOT/outputs/pythia-1.4b/seed_reps/seed_${SEED}"
  TEACHERS_DIR="$OUT_ROOT/teachers"
  STUDENTS_DIR="$OUT_ROOT/students"
  MIA_DIR="$OUT_ROOT/mia"
  mkdir -p "$TEACHERS_DIR" "$STUDENTS_DIR" "$MIA_DIR"

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
    $PY - <<PY
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
        # Resume without optimizer: load weights from checkpoint and start a fresh optimizer.
        chosen_model="$ckpt_dir"
      fi
    fi
    printf -v "$model_out" "%s" "$chosen_model"
    printf -v "$resume_out" "%s" "$resume_arg"
  }

  model_ready() {
    local dir="$1"
    [[ -f "$dir/model.safetensors" || -f "$dir/pytorch_model.bin" || -f "$dir/model.safetensors.index.json" || -f "$dir/model-00001-of-00002.safetensors" ]]
  }

  TOTAL_C1="$(total_steps "$TEACHER_C1_DATA" "$EPOCHS" "$PER_DEVICE_BATCH" "$GRAD_ACCUM")"
  TOTAL_C3="$(total_steps "$TEACHER_C3_DATA" "$EPOCHS" "$PER_DEVICE_BATCH" "$GRAD_ACCUM")"
  TOTAL_DISTILL="$(total_steps "$DATASETS_DIR/distill" "$EPOCHS" "$PER_DEVICE_BATCH" "$GRAD_ACCUM")"

  echo "[seed-reps] Train C1 teacher"
  TEACHER_MODEL_C1="$TEACHERS_DIR/c1"
  if model_ready "$TEACHERS_DIR/c1"; then
    echo "[seed-reps] Skip C1 (model exists)"
  else
    RESUME_C1="$(latest_ckpt "$TEACHERS_DIR/c1")"
    MODEL_C1="$MODEL"
    RESUME_ARG_C1=""
    MAX_STEPS_ARG_C1=""
    WARMUP_ARG_C1="--warmup-steps $WARMUP_STEPS"
    if [[ -n "$RESUME_C1" ]]; then
      STEP_C1="$(ckpt_step "$RESUME_C1")"
      REMAIN_C1=$((TOTAL_C1 - STEP_C1))
      if (( REMAIN_C1 <= 0 )); then
        echo "[seed-reps] C1 already complete at $RESUME_C1"
        TEACHER_MODEL_C1="$RESUME_C1"
      else
        pick_model_resume "$MODEL" "$RESUME_C1" MODEL_C1 RESUME_ARG_C1
        if [[ -z "$RESUME_ARG_C1" ]]; then
          MAX_STEPS_ARG_C1="--max-steps $REMAIN_C1"
          WARMUP_ARG_C1="--warmup-steps 0"
        fi
        CUDA_VISIBLE_DEVICES="$TRAIN_GPU" $PY "$ROOT/src/train_teacher.py" \
          --model "$MODEL_C1" \
          --dataset "$TEACHER_C1_DATA" \
          --output "$TEACHERS_DIR/c1" \
          --max-length "$MAX_LENGTH" \
          --epochs "$EPOCHS" \
          --lr "$LR" \
          $WARMUP_ARG_C1 \
          $MAX_STEPS_ARG_C1 \
          --per-device-batch "$PER_DEVICE_BATCH" \
          --grad-accum "$GRAD_ACCUM" \
          --optim "$OPTIM" \
          --seed "$SEED" \
          $RESUME_ARG_C1 \
          $BF16_FLAG
      fi
    else
      CUDA_VISIBLE_DEVICES="$TRAIN_GPU" $PY "$ROOT/src/train_teacher.py" \
        --model "$MODEL_C1" \
        --dataset "$TEACHER_C1_DATA" \
        --output "$TEACHERS_DIR/c1" \
        --max-length "$MAX_LENGTH" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --warmup-steps "$WARMUP_STEPS" \
        --per-device-batch "$PER_DEVICE_BATCH" \
        --grad-accum "$GRAD_ACCUM" \
        --optim "$OPTIM" \
        --seed "$SEED" \
        $RESUME_ARG_C1 \
        $BF16_FLAG
    fi
  fi
  if model_ready "$TEACHERS_DIR/c1"; then
    TEACHER_MODEL_C1="$TEACHERS_DIR/c1"
  elif [[ -n "${RESUME_C1:-}" ]]; then
    TEACHER_MODEL_C1="$RESUME_C1"
  fi

  echo "[seed-reps] Train C3 teacher"
  TEACHER_MODEL_C3="$TEACHERS_DIR/c3"
  if model_ready "$TEACHERS_DIR/c3"; then
    echo "[seed-reps] Skip C3 (model exists)"
  else
    RESUME_C3="$(latest_ckpt "$TEACHERS_DIR/c3")"
    MODEL_C3="$MODEL"
    RESUME_ARG_C3=""
    MAX_STEPS_ARG_C3=""
    WARMUP_ARG_C3="--warmup-steps $WARMUP_STEPS"
    if [[ -n "$RESUME_C3" ]]; then
      STEP_C3="$(ckpt_step "$RESUME_C3")"
      REMAIN_C3=$((TOTAL_C3 - STEP_C3))
      if (( REMAIN_C3 <= 0 )); then
        echo "[seed-reps] C3 already complete at $RESUME_C3"
        TEACHER_MODEL_C3="$RESUME_C3"
      else
        pick_model_resume "$MODEL" "$RESUME_C3" MODEL_C3 RESUME_ARG_C3
        if [[ -z "$RESUME_ARG_C3" ]]; then
          MAX_STEPS_ARG_C3="--max-steps $REMAIN_C3"
          WARMUP_ARG_C3="--warmup-steps 0"
        fi
        CUDA_VISIBLE_DEVICES="$TRAIN_GPU" $PY "$ROOT/src/train_teacher.py" \
          --model "$MODEL_C3" \
          --dataset "$TEACHER_C3_DATA" \
          --output "$TEACHERS_DIR/c3" \
          --max-length "$MAX_LENGTH" \
          --epochs "$EPOCHS" \
          --lr "$LR" \
          $WARMUP_ARG_C3 \
          $MAX_STEPS_ARG_C3 \
          --per-device-batch "$PER_DEVICE_BATCH" \
          --grad-accum "$GRAD_ACCUM" \
          --optim "$OPTIM" \
          --seed "$SEED" \
          $RESUME_ARG_C3 \
          $BF16_FLAG
      fi
    else
      CUDA_VISIBLE_DEVICES="$TRAIN_GPU" $PY "$ROOT/src/train_teacher.py" \
        --model "$MODEL_C3" \
        --dataset "$TEACHER_C3_DATA" \
        --output "$TEACHERS_DIR/c3" \
        --max-length "$MAX_LENGTH" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --warmup-steps "$WARMUP_STEPS" \
        --per-device-batch "$PER_DEVICE_BATCH" \
        --grad-accum "$GRAD_ACCUM" \
        --optim "$OPTIM" \
        --seed "$SEED" \
        $RESUME_ARG_C3 \
        $BF16_FLAG
    fi
  fi
  if model_ready "$TEACHERS_DIR/c3"; then
    TEACHER_MODEL_C3="$TEACHERS_DIR/c3"
  elif [[ -n "${RESUME_C3:-}" ]]; then
    TEACHER_MODEL_C3="$RESUME_C3"
  fi

  echo "[seed-reps] Distill C1 student"
  if model_ready "$STUDENTS_DIR/c1"; then
    echo "[seed-reps] Skip C1 distill (model exists)"
  else
    RESUME_DC1="$(latest_ckpt "$STUDENTS_DIR/c1")"
    STUDENT_C1="$STUDENT"
    RESUME_ARG_DC1=""
    MAX_STEPS_ARG_DC1=""
    WARMUP_ARG_DC1="--warmup-steps $WARMUP_STEPS"
    if [[ -n "$RESUME_DC1" && -d "$RESUME_DC1" ]]; then
      if [[ -f "$RESUME_DC1/optimizer.pt" ]]; then
        RESUME_ARG_DC1="--resume $RESUME_DC1"
      else
        STUDENT_C1="$RESUME_DC1"
        STEP_DC1="$(ckpt_step "$RESUME_DC1")"
        REMAIN_DC1=$((TOTAL_DISTILL - STEP_DC1))
        if (( REMAIN_DC1 > 0 )); then
          MAX_STEPS_ARG_DC1="--max-steps $REMAIN_DC1"
          WARMUP_ARG_DC1="--warmup-steps 0"
        fi
      fi
    fi
    CUDA_VISIBLE_DEVICES="$DISTILL_GPU" $PY "$ROOT/src/distill_student.py" \
      --teacher "$TEACHER_MODEL_C1" \
      --student "$STUDENT_C1" \
      --dataset "$DATASETS_DIR/distill" \
      --output "$STUDENTS_DIR/c1" \
      --max-length "$MAX_LENGTH" \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      $WARMUP_ARG_DC1 \
      $MAX_STEPS_ARG_DC1 \
      --per-device-batch "$PER_DEVICE_BATCH" \
      --grad-accum "$GRAD_ACCUM" \
      --optim "$OPTIM" \
      --seed "$SEED" \
      $RESUME_ARG_DC1 \
      $BF16_FLAG
  fi

  echo "[seed-reps] Distill C3 student"
  if model_ready "$STUDENTS_DIR/c3"; then
    echo "[seed-reps] Skip C3 distill (model exists)"
  else
    RESUME_DC3="$(latest_ckpt "$STUDENTS_DIR/c3")"
    STUDENT_C3="$STUDENT"
    RESUME_ARG_DC3=""
    MAX_STEPS_ARG_DC3=""
    WARMUP_ARG_DC3="--warmup-steps $WARMUP_STEPS"
    if [[ -n "$RESUME_DC3" && -d "$RESUME_DC3" ]]; then
      if [[ -f "$RESUME_DC3/optimizer.pt" ]]; then
        RESUME_ARG_DC3="--resume $RESUME_DC3"
      else
        STUDENT_C3="$RESUME_DC3"
        STEP_DC3="$(ckpt_step "$RESUME_DC3")"
        REMAIN_DC3=$((TOTAL_DISTILL - STEP_DC3))
        if (( REMAIN_DC3 > 0 )); then
          MAX_STEPS_ARG_DC3="--max-steps $REMAIN_DC3"
          WARMUP_ARG_DC3="--warmup-steps 0"
        fi
      fi
    fi
    CUDA_VISIBLE_DEVICES="$DISTILL_GPU" $PY "$ROOT/src/distill_student.py" \
      --teacher "$TEACHER_MODEL_C3" \
      --student "$STUDENT_C3" \
      --dataset "$DATASETS_DIR/distill" \
      --output "$STUDENTS_DIR/c3" \
      --max-length "$MAX_LENGTH" \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      $WARMUP_ARG_DC3 \
      $MAX_STEPS_ARG_DC3 \
      --per-device-batch "$PER_DEVICE_BATCH" \
      --grad-accum "$GRAD_ACCUM" \
      --optim "$OPTIM" \
      --seed "$SEED" \
      $RESUME_ARG_DC3 \
      $BF16_FLAG
  fi

  echo "[seed-reps] C5m unlearning"
  if model_ready "$TEACHERS_DIR/c5m_unlearn"; then
    echo "[seed-reps] Skip C5m unlearn (model exists)"
  else
    CUDA_VISIBLE_DEVICES="$UNLEARN_GPU" $PY "$ROOT/src/unlearn_teacher.py" \
      --model "$TEACHER_MODEL_C1" \
      --forget-dataset "$FORGET_DATA" \
      --retain-dataset "$RETAIN_DATA" \
      --output "$TEACHERS_DIR/c5m_unlearn" \
      $BF16_FLAG \
      --optim "$OPTIM" \
      --batch-size "$PER_DEVICE_BATCH" \
      --grad-accum "$GRAD_ACCUM" \
      --epochs 1 \
      --lr "$LR" \
      --alpha "$C5_ALPHA" \
      --beta "$C5_BETA" \
      --kl-model "$C5_KL_MODEL" \
      --kl-weight "$C5_KL_WEIGHT" \
      --kl-device "$C5_KL_DEVICE" \
      --kl-every "$C5_KL_EVERY" \
      --early-stop-patience "$C5_EARLY_PATIENCE" \
      --early-stop-min-steps "$C5_EARLY_MIN_STEPS" \
      --early-stop-eval-every "$C5_EARLY_EVAL_EVERY" \
      --seed "$SEED"
  fi

  echo "[seed-reps] Distill C5m student"
  if model_ready "$STUDENTS_DIR/c5m"; then
    echo "[seed-reps] Skip C5m distill (model exists)"
  else
    RESUME_DC5M="$(latest_ckpt "$STUDENTS_DIR/c5m")"
    STUDENT_C5M="$STUDENT"
    RESUME_ARG_DC5M=""
    MAX_STEPS_ARG_DC5M=""
    WARMUP_ARG_DC5M="--warmup-steps $WARMUP_STEPS"
    if [[ -n "$RESUME_DC5M" && -d "$RESUME_DC5M" ]]; then
      if [[ -f "$RESUME_DC5M/optimizer.pt" ]]; then
        RESUME_ARG_DC5M="--resume $RESUME_DC5M"
      else
        STUDENT_C5M="$RESUME_DC5M"
        STEP_DC5M="$(ckpt_step "$RESUME_DC5M")"
        REMAIN_DC5M=$((TOTAL_DISTILL - STEP_DC5M))
        if (( REMAIN_DC5M > 0 )); then
          MAX_STEPS_ARG_DC5M="--max-steps $REMAIN_DC5M"
          WARMUP_ARG_DC5M="--warmup-steps 0"
        fi
      fi
    fi
    CUDA_VISIBLE_DEVICES="$DISTILL_GPU" $PY "$ROOT/src/distill_student.py" \
      --teacher "$TEACHERS_DIR/c5m_unlearn" \
      --student "$STUDENT_C5M" \
      --dataset "$DATASETS_DIR/distill" \
      --output "$STUDENTS_DIR/c5m" \
      --max-length "$MAX_LENGTH" \
      --epochs "$EPOCHS" \
      --lr "$LR" \
      $WARMUP_ARG_DC5M \
      $MAX_STEPS_ARG_DC5M \
      --per-device-batch "$PER_DEVICE_BATCH" \
      --grad-accum "$GRAD_ACCUM" \
      --optim "$OPTIM" \
      --seed "$SEED" \
      $RESUME_ARG_DC5M \
      $BF16_FLAG
  fi

  echo "[seed-reps] Eval MIA (C1, C3, C5m student + C5m teacher)"
  CUDA_VISIBLE_DEVICES="$DISTILL_GPU" $PY "$ROOT/src/eval_mia.py" \
    --model "$STUDENTS_DIR/c1" \
    --target-holdout "$DATASETS_DIR/eval_target_holdout" \
    --nonmember "$DATASETS_DIR/eval_nonmember" \
    --holdout-map "$DATASETS_DIR/holdout_map.json" \
    --output "$MIA_DIR/c1_student.json" \
    --batch-size 4 \
    $BF16_FLAG

  CUDA_VISIBLE_DEVICES="$DISTILL_GPU" $PY "$ROOT/src/eval_mia.py" \
    --model "$STUDENTS_DIR/c3" \
    --target-holdout "$DATASETS_DIR/eval_target_holdout" \
    --nonmember "$DATASETS_DIR/eval_nonmember" \
    --holdout-map "$DATASETS_DIR/holdout_map.json" \
    --output "$MIA_DIR/c3_student.json" \
    --batch-size 4 \
    $BF16_FLAG

  CUDA_VISIBLE_DEVICES="$DISTILL_GPU" $PY "$ROOT/src/eval_mia.py" \
    --model "$STUDENTS_DIR/c5m" \
    --target-holdout "$DATASETS_DIR/eval_target_holdout" \
    --nonmember "$DATASETS_DIR/eval_nonmember" \
    --holdout-map "$DATASETS_DIR/holdout_map.json" \
    --output "$MIA_DIR/c5m_student.json" \
    --batch-size 4 \
    $BF16_FLAG

  CUDA_VISIBLE_DEVICES="$DISTILL_GPU" $PY "$ROOT/src/eval_mia.py" \
    --model "$TEACHERS_DIR/c5m_unlearn" \
    --target-holdout "$DATASETS_DIR/eval_target_holdout" \
    --nonmember "$DATASETS_DIR/eval_nonmember" \
    --holdout-map "$DATASETS_DIR/holdout_map.json" \
    --output "$MIA_DIR/c5m_teacher.json" \
    --batch-size 4 \
    $BF16_FLAG

  echo "[seed-reps] Completed seed $SEED"
done

echo "[seed-reps] All seeds complete."
