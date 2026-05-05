#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
LOG="$ROOT/outputs/logs/c5sweep_distill.log"
DATASETS="$ROOT/data/datasets/pythia-1.4b"
OUT_ROOT="$ROOT/outputs/pythia-1.4b/c5sweep"
STUDENT_MODEL="EleutherAI/pythia-410m"

GPU_ID="${GPU_ID:-3}"

for A in 0.3 0.5 0.7; do
  OUT="$OUT_ROOT/alpha_${A}"
  TEACHER="$OUT/teacher_unlearn"
  STUDENT_OUT="$OUT/student"
  MIA_OUT="$OUT/mia_student.json"

  if [[ ! -f "$TEACHER/model.safetensors" && ! -f "$TEACHER/model-00001-of-00002.safetensors" ]]; then
    echo "[c5sweep-distill] alpha=$A missing teacher at $TEACHER" | tee -a "$LOG"
    continue
  fi

  echo "[c5sweep-distill] alpha=$A start $(date)" | tee -a "$LOG"

  if [[ -f "$STUDENT_OUT/model.safetensors" || -f "$STUDENT_OUT/pytorch_model.bin" ]]; then
    echo "[c5sweep-distill] alpha=$A student exists, skip distill" | tee -a "$LOG"
  else
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" "$ROOT/src/distill_student.py" \
      --teacher "$TEACHER" \
      --student "$STUDENT_MODEL" \
      --dataset "$DATASETS/distill" \
      --output "$STUDENT_OUT" \
      --max-length 512 \
      --epochs 3 \
      --lr 2e-5 \
      --warmup-steps 500 \
      --per-device-batch 2 \
      --grad-accum 16 \
      --optim adamw_8bit \
      --seed 13 \
      --bf16 \
      |& tee -a "$LOG"
  fi

  if [[ -f "$MIA_OUT" ]]; then
    echo "[c5sweep-distill] alpha=$A eval exists, skip" | tee -a "$LOG"
  else
    CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY" "$ROOT/src/eval_mia.py" \
      --model "$STUDENT_OUT" \
      --target-holdout "$DATASETS/eval_target_holdout" \
      --nonmember "$DATASETS/eval_nonmember" \
      --holdout-map "$DATASETS/holdout_map.json" \
      --output "$MIA_OUT" \
      --batch-size 4 \
      --bf16 \
      |& tee -a "$LOG"
  fi

done

