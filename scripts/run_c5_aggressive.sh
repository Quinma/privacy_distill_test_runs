#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

MODEL="${MODEL:-EleutherAI/pythia-1.4b}"
STUDENT="${STUDENT:-EleutherAI/pythia-410m}"
RUN_TAG="${RUN_TAG:-$(echo "$MODEL" | sed 's#.*/##')}"

DATASETS_DIR="${DATASETS_DIR:-$ROOT/data/datasets/$RUN_TAG}"
TEACHERS_DIR="${TEACHERS_DIR:-$ROOT/outputs/$RUN_TAG/teachers}"
STUDENTS_DIR="${STUDENTS_DIR:-$ROOT/outputs/$RUN_TAG/students}"
MIA_DIR="${MIA_DIR:-$ROOT/outputs/$RUN_TAG/mia}"

FORGET_DATA="${FORGET_DATA:-$DATASETS_DIR/target_train}"
RETAIN_DATA="${RETAIN_DATA:-$DATASETS_DIR/teacher_c2}"

UNLEARN_OUT="${UNLEARN_OUT:-$TEACHERS_DIR/c5_unlearn}"
STUDENT_OUT="${STUDENT_OUT:-$STUDENTS_DIR/c5}"

MAX_LENGTH="${MAX_LENGTH:-512}"
BF16="${BF16:-1}"

UNLEARN_LR="${UNLEARN_LR:-2e-5}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-1}"
ALPHA="${ALPHA:-1.0}"
BETA="${BETA:-1.0}"
UNLEARN_OPTIM="${UNLEARN_OPTIM:-adamw_8bit}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"

VISIBLE_GPUS="${VISIBLE_GPUS:-0,1,2,3}"
UNLEARN_FSDP_NPROC="${UNLEARN_FSDP_NPROC:-1}"
UNLEARN_GPU="${UNLEARN_GPU:-0}"
DISTILL_DDP_NPROC="${DISTILL_DDP_NPROC:-1}"
DISTILL_GPU="${DISTILL_GPU:-0}"

DISTILL_EPOCHS="${DISTILL_EPOCHS:-1}"
DISTILL_LR="${DISTILL_LR:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
DISTILL_OPTIM="${DISTILL_OPTIM:-adamw_8bit}"

EVAL_GPU="${EVAL_GPU:-0}"
RUN_STATS="${RUN_STATS:-0}"

BF16_FLAG=""
if [[ "$BF16" == "1" ]]; then
  BF16_FLAG="--bf16"
fi

mkdir -p "$TEACHERS_DIR" "$STUDENTS_DIR" "$MIA_DIR"

echo "[c5] Aggressive unlearning (alpha=$ALPHA beta=$BETA) on $MODEL"
if [[ "$UNLEARN_FSDP_NPROC" -gt 1 ]]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" \
    $PY -m torch.distributed.run --nproc_per_node="$UNLEARN_FSDP_NPROC" "$ROOT/src/unlearn_teacher.py" \
      --model "$TEACHERS_DIR/c1" \
      --forget-dataset "$FORGET_DATA" \
      --retain-dataset "$RETAIN_DATA" \
      --output "$UNLEARN_OUT" \
      $BF16_FLAG \
      --optim "$UNLEARN_OPTIM" \
      --batch-size "$PER_DEVICE_BATCH" \
      --grad-accum "$GRAD_ACCUM" \
      --epochs "$UNLEARN_EPOCHS" \
      --lr "$UNLEARN_LR" \
      --alpha "$ALPHA" \
      --beta "$BETA" \
      --fsdp \
      --cpu-offload
else
  CUDA_VISIBLE_DEVICES="$UNLEARN_GPU" \
    $PY "$ROOT/src/unlearn_teacher.py" \
      --model "$TEACHERS_DIR/c1" \
      --forget-dataset "$FORGET_DATA" \
      --retain-dataset "$RETAIN_DATA" \
      --output "$UNLEARN_OUT" \
      $BF16_FLAG \
      --optim "$UNLEARN_OPTIM" \
      --batch-size "$PER_DEVICE_BATCH" \
      --grad-accum "$GRAD_ACCUM" \
      --epochs "$UNLEARN_EPOCHS" \
      --lr "$UNLEARN_LR" \
      --alpha "$ALPHA" \
      --beta "$BETA"
fi

echo "[c5] Distilling student from aggressive unlearned teacher"
if [[ "$DISTILL_DDP_NPROC" -gt 1 ]]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" \
    $PY -m torch.distributed.run --nproc_per_node="$DISTILL_DDP_NPROC" "$ROOT/src/distill_student.py" \
      --teacher "$UNLEARN_OUT" \
      --student "$STUDENT" \
      --dataset "$DATASETS_DIR/distill" \
      --output "$STUDENT_OUT" \
      --max-length "$MAX_LENGTH" \
      --epochs "$DISTILL_EPOCHS" \
      --lr "$DISTILL_LR" \
      --warmup-steps "$WARMUP_STEPS" \
      --per-device-batch "$PER_DEVICE_BATCH" \
      --grad-accum "$GRAD_ACCUM" \
      --optim "$DISTILL_OPTIM" \
      $BF16_FLAG
else
  CUDA_VISIBLE_DEVICES="$DISTILL_GPU" \
    $PY "$ROOT/src/distill_student.py" \
      --teacher "$UNLEARN_OUT" \
      --student "$STUDENT" \
      --dataset "$DATASETS_DIR/distill" \
      --output "$STUDENT_OUT" \
      --max-length "$MAX_LENGTH" \
      --epochs "$DISTILL_EPOCHS" \
      --lr "$DISTILL_LR" \
      --warmup-steps "$WARMUP_STEPS" \
      --per-device-batch "$PER_DEVICE_BATCH" \
      --grad-accum "$GRAD_ACCUM" \
      --optim "$DISTILL_OPTIM" \
      $BF16_FLAG
fi

echo "[c5] Evaluating MIA and utility"
CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
  $PY "$ROOT/src/eval_mia.py" \
    --model "$STUDENT_OUT" \
    --target-holdout "$DATASETS_DIR/eval_target_holdout" \
    --nonmember "$DATASETS_DIR/eval_nonmember" \
    --holdout-map "$DATASETS_DIR/holdout_map.json" \
    --output "$MIA_DIR/c5_student.json" \
    --batch-size 4 \
    --max-length "$MAX_LENGTH" \
    $BF16_FLAG

CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
  $PY "$ROOT/src/eval_mia.py" \
    --model "$UNLEARN_OUT" \
    --target-holdout "$DATASETS_DIR/eval_target_holdout" \
    --nonmember "$DATASETS_DIR/eval_nonmember" \
    --holdout-map "$DATASETS_DIR/holdout_map.json" \
    --output "$MIA_DIR/c5_teacher.json" \
    --batch-size 4 \
    --max-length "$MAX_LENGTH" \
    $BF16_FLAG

CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
  $PY "$ROOT/src/eval_ppl.py" \
    --model "$STUDENT_OUT" \
    --dataset "$DATASETS_DIR/distill" \
    --output "$MIA_DIR/utility_c5_student.json" \
    --batch-size 4 \
    --max-samples 500 \
    $BF16_FLAG

CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
  $PY "$ROOT/src/eval_ppl.py" \
    --model "$UNLEARN_OUT" \
    --dataset "$DATASETS_DIR/distill" \
    --output "$MIA_DIR/utility_c5_teacher.json" \
    --batch-size 4 \
    --max-samples 500 \
    $BF16_FLAG

CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
  $PY "$ROOT/src/eval_ppl.py" \
    --model "$STUDENT_OUT" \
    --dataset "$DATASETS_DIR/eval_target_holdout_tok" \
    --output "$MIA_DIR/utility_c5_student_holdout.json" \
    --batch-size 4 \
    --max-samples 0 \
    $BF16_FLAG

CUDA_VISIBLE_DEVICES="$EVAL_GPU" \
  $PY "$ROOT/src/eval_ppl.py" \
    --model "$UNLEARN_OUT" \
    --dataset "$DATASETS_DIR/eval_target_holdout_tok" \
    --output "$MIA_DIR/utility_c5_teacher_holdout.json" \
    --batch-size 4 \
    --max-samples 0 \
    $BF16_FLAG

if [[ "$RUN_STATS" == "1" ]]; then
  if [[ -f "$MIA_DIR/c1_student.json" && -f "$MIA_DIR/c2_student.json" && -f "$MIA_DIR/c3_student.json" && -f "$MIA_DIR/c4_teacher.json" ]]; then
    $PY "$ROOT/src/compute_stats.py" \
      --c1 "$MIA_DIR/c1_student.json" \
      --c2 "$MIA_DIR/c2_student.json" \
      --c3 "$MIA_DIR/c3_student.json" \
      --c4 "$MIA_DIR/c4_teacher.json" \
      --c5 "$MIA_DIR/c5_student.json" \
      --out-dir "$MIA_DIR"
  fi
fi

echo "[c5] Done"
