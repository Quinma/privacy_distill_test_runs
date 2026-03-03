#!/usr/bin/env bash
set -euo pipefail

OPTIM=${OPTIM:-adamw_torch}
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-8}
EPOCHS=${EPOCHS:-1}
WORKERS=${WORKERS:-4}
# Use 8-bit AdamW if requested (requires bitsandbytes).
# Example: OPTIM=adamw_8bit bash scripts/run_train_teachers.sh

# C1 teacher (retain + target) on GPU0
CUDA_VISIBLE_DEVICES=0 python -u src/train_teacher.py \
  --dataset data/datasets/teacher_c1 \
  --output outputs/teachers/c1 \
  --per-device-batch "$BATCH" \
  --grad-accum "$ACCUM" \
  --epochs "$EPOCHS" \
  --bf16 \
  --optim "$OPTIM" \
  --dataloader-num-workers "$WORKERS" \
  --dataloader-pin-memory \
  --seed 13 &

# C2 teacher (retrained, retain only) on GPU1
CUDA_VISIBLE_DEVICES=1 python -u src/train_teacher.py \
  --dataset data/datasets/teacher_c2 \
  --output outputs/teachers/c2 \
  --per-device-batch "$BATCH" \
  --grad-accum "$ACCUM" \
  --epochs "$EPOCHS" \
  --bf16 \
  --optim "$OPTIM" \
  --dataloader-num-workers "$WORKERS" \
  --dataloader-pin-memory \
  --seed 17 &

# C3 teacher (clean baseline) on GPU2
CUDA_VISIBLE_DEVICES=2 python -u src/train_teacher.py \
  --dataset data/datasets/teacher_c3 \
  --output outputs/teachers/c3 \
  --per-device-batch "$BATCH" \
  --grad-accum "$ACCUM" \
  --epochs "$EPOCHS" \
  --bf16 \
  --optim "$OPTIM" \
  --dataloader-num-workers "$WORKERS" \
  --dataloader-pin-memory \
  --seed 19 &

wait
