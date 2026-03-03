#!/usr/bin/env bash
set -euo pipefail

OPTIM=${OPTIM:-adamw_torch}
BATCH=${BATCH:-4}
ACCUM=${ACCUM:-8}
EPOCHS=${EPOCHS:-1}
WORKERS=${WORKERS:-4}
USE_GC=${USE_GC:-0}
GC_FLAG=""
if [[ "$USE_GC" == "1" ]]; then
  GC_FLAG="--grad-checkpointing"
fi

# C1 student (dirty teacher) on GPU0
CUDA_VISIBLE_DEVICES=0 python -u src/distill_student.py \
  --teacher outputs/teachers/c1 \
  --dataset data/datasets/distill \
  --output outputs/students/c1 \
  --per-device-batch "$BATCH" \
  --grad-accum "$ACCUM" \
  --epochs "$EPOCHS" \
  --bf16 \
  --optim "$OPTIM" \
  --dataloader-num-workers "$WORKERS" \
  --dataloader-pin-memory \
  $GC_FLAG \
  --temperature 2.0 \
  --alpha 0.5 \
  --seed 13 &

# C2 student (retrained teacher) on GPU1
CUDA_VISIBLE_DEVICES=1 python -u src/distill_student.py \
  --teacher outputs/teachers/c2 \
  --dataset data/datasets/distill \
  --output outputs/students/c2 \
  --per-device-batch "$BATCH" \
  --grad-accum "$ACCUM" \
  --epochs "$EPOCHS" \
  --bf16 \
  --optim "$OPTIM" \
  --dataloader-num-workers "$WORKERS" \
  --dataloader-pin-memory \
  $GC_FLAG \
  --temperature 2.0 \
  --alpha 0.5 \
  --seed 17 &

# C3 student (clean teacher) on GPU2
CUDA_VISIBLE_DEVICES=2 python -u src/distill_student.py \
  --teacher outputs/teachers/c3 \
  --dataset data/datasets/distill \
  --output outputs/students/c3 \
  --per-device-batch "$BATCH" \
  --grad-accum "$ACCUM" \
  --epochs "$EPOCHS" \
  --bf16 \
  --optim "$OPTIM" \
  --dataloader-num-workers "$WORKERS" \
  --dataloader-pin-memory \
  $GC_FLAG \
  --temperature 2.0 \
  --alpha 0.5 \
  --seed 19 &

wait
