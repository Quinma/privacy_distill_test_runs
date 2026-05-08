#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
MODEL="${MODEL:-EleutherAI/pythia-1.4b}"
RUN_TAG="${RUN_TAG:-$(basename "$MODEL")}" 
DATASETS_DIR="${DATASETS_DIR:-$ROOT/data/datasets/$RUN_TAG}"
TEACHERS_DIR="${TEACHERS_DIR:-$ROOT/outputs/$RUN_TAG/teachers}"

OPTIM="${OPTIM:-adamw_torch}"
BATCH="${TRAIN_BATCH:-2}"
ACCUM="${TRAIN_ACCUM:-16}"
EPOCHS="${TRAIN_EPOCHS:-3}"
LR="${TRAIN_LR:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
WORKERS="${WORKERS:-4}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BF16="${BF16:-1}"
USE_GC="${TRAIN_GRAD_CHECKPOINTING:-0}"

RUN_C1="${RUN_C1:-1}"
RUN_C2="${RUN_C2:-1}"
RUN_C3="${RUN_C3:-1}"
TRAIN_GPU_C1="${TRAIN_GPU_C1:-0}"
TRAIN_GPU_C2="${TRAIN_GPU_C2:-1}"
TRAIN_GPU_C3="${TRAIN_GPU_C3:-2}"
SEED_C1="${SEED_C1:-13}"
SEED_C2="${SEED_C2:-17}"
SEED_C3="${SEED_C3:-19}"

mkdir -p "$TEACHERS_DIR"

common_args=(
  --model "$MODEL"
  --max-length "$MAX_LENGTH"
  --per-device-batch "$BATCH"
  --grad-accum "$ACCUM"
  --epochs "$EPOCHS"
  --lr "$LR"
  --warmup-steps "$WARMUP_STEPS"
  --optim "$OPTIM"
  --dataloader-num-workers "$WORKERS"
  --dataloader-pin-memory
)
[[ "$BF16" == "1" ]] && common_args+=(--bf16)
[[ "$USE_GC" == "1" ]] && common_args+=(--grad-checkpointing)

run_cond() {
  local condition="$1" gpu="$2" seed="$3"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u "$ROOT/src/train_teacher.py" \
    --dataset "$DATASETS_DIR/teacher_$condition" \
    --output "$TEACHERS_DIR/$condition" \
    --seed "$seed" \
    "${common_args[@]}"
}

jobs=0
[[ "$RUN_C1" == "1" ]] && run_cond c1 "$TRAIN_GPU_C1" "$SEED_C1" & jobs=$((jobs+1))
[[ "$RUN_C2" == "1" ]] && run_cond c2 "$TRAIN_GPU_C2" "$SEED_C2" & jobs=$((jobs+1))
[[ "$RUN_C3" == "1" ]] && run_cond c3 "$TRAIN_GPU_C3" "$SEED_C3" & jobs=$((jobs+1))
[[ "$jobs" -eq 0 ]] && { echo "ERROR: no teacher conditions enabled" >&2; exit 1; }
wait
