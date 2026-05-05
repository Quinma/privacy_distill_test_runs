#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

MODEL="${MODEL:-EleutherAI/pythia-1.4b}"
STUDENT="${STUDENT:-EleutherAI/pythia-410m}"
RUN_TAG="${RUN_TAG:-$(echo "$MODEL" | sed 's#.*/##')}"

DATASET="${DATASET:-bradfordlevy/BeanCounter}"
CONFIG="${CONFIG:-clean}"
REVISION="${REVISION:-}"
SPLIT="${SPLIT:-train}"
FORM_TYPES="${FORM_TYPES:-10-K}"
TOKENIZER="${TOKENIZER:-$MODEL}"
STATS_PATH="${STATS_PATH:-$ROOT/data/bean_counter_stats.jsonl}"
CIK_MAP="${CIK_MAP:-$ROOT/data/sec_index_10k.jsonl}"

DATASETS_DIR="${DATASETS_DIR:-$ROOT/data/datasets/$RUN_TAG}"
TEACHERS_DIR="${TEACHERS_DIR:-$ROOT/outputs/$RUN_TAG/teachers}"
STUDENTS_DIR="${STUDENTS_DIR:-$ROOT/outputs/$RUN_TAG/students}"
MIA_DIR="${MIA_DIR:-$ROOT/outputs/$RUN_TAG/mia}"

FORGET_OUT="${FORGET_OUT:-$DATASETS_DIR/random_forget_train}"
FORGET_CIKS="${FORGET_CIKS:-$DATASETS_DIR/c5r_forget_ciks.json}"

VISIBLE_GPUS="${VISIBLE_GPUS:-0,1,2,3}"
NPROC="${NPROC:-4}"
UNLEARN_NPROC="${UNLEARN_NPROC:-$NPROC}"
DISTILL_NPROC="${DISTILL_NPROC:-$NPROC}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"

MAX_LENGTH="${MAX_LENGTH:-512}"
MAX_TOKENS_PER_COMPANY="${MAX_TOKENS_PER_COMPANY:-0}"
MIN_TOKENS="${MIN_TOKENS:-200000}"
NUM_FORGET="${NUM_FORGET:-50}"
SEED="${SEED:-13}"

DISTILL_EPOCHS="${DISTILL_EPOCHS:-3}"
DISTILL_OPTIM="${DISTILL_OPTIM:-adamw_8bit}"
DISTILL_LR="${DISTILL_LR:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
DISTILL_GRAD_CHECKPOINTING="${DISTILL_GRAD_CHECKPOINTING:-0}"
DISTILL_TEACHER_DEVICE="${DISTILL_TEACHER_DEVICE:-cuda}"

UNLEARN_LR="${UNLEARN_LR:-2e-5}"
ALPHA="${ALPHA:-1.0}"
BETA="${BETA:-1.0}"

BF16_FLAG="--bf16"
CLEAN="${CLEAN:-0}"

DISTILL_GRAD_CHECKPOINTING_FLAG=""
if [[ "$DISTILL_GRAD_CHECKPOINTING" == "1" ]]; then
  DISTILL_GRAD_CHECKPOINTING_FLAG="--grad-checkpointing"
fi

mkdir -p "$TEACHERS_DIR" "$STUDENTS_DIR" "$MIA_DIR"

normalize_visible_gpus() {
  local required="$1"
  local current="${VISIBLE_GPUS:-}"
  local count=0
  if [[ -n "$current" ]]; then
    count="$(awk -F',' '{print NF}' <<< "$current")"
  fi
  if [[ -z "$current" || "$count" -lt "$required" ]]; then
    seq -s, 0 $((required-1))
  else
    echo "$current"
  fi
}

MAX_NPROC="$UNLEARN_NPROC"
if [[ "$DISTILL_NPROC" -gt "$MAX_NPROC" ]]; then MAX_NPROC="$DISTILL_NPROC"; fi
VISIBLE_GPUS="$(normalize_visible_gpus "$MAX_NPROC")"

normalize_visible_gpus() {
  local required="$1"
  local current="${VISIBLE_GPUS:-}"
  local count=0
  if [[ -n "$current" ]]; then
    count="$(awk -F',' '{print NF}' <<< "$current")"
  fi
  if [[ -z "$current" || "$count" -lt "$required" ]]; then
    seq -s, 0 $((required-1))
  else
    echo "$current"
  fi
}

VISIBLE_GPUS="$(normalize_visible_gpus "$NPROC")"

BNB_AVAILABLE=0
if "$PY" -c "import bitsandbytes" >/dev/null 2>&1; then
  BNB_AVAILABLE=1
fi

OPTIM_TORCH_FALLBACK="$("$PY" - <<'PY'
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

if [[ "$DISTILL_OPTIM" == "adamw_8bit" && "$BNB_AVAILABLE" != "1" ]]; then
  echo "[c5r] WARN: bitsandbytes not available; falling back from DISTILL_OPTIM=adamw_8bit to $OPTIM_TORCH_FALLBACK" >&2
  DISTILL_OPTIM="$OPTIM_TORCH_FALLBACK"
fi

echo "[c5r] Building random forget dataset..."
$PY "$ROOT/src/build_random_forget.py" \
  --dataset "$DATASET" \
  --config "$CONFIG" \
  ${REVISION:+--revision "$REVISION"} \
  --split "$SPLIT" \
  --cik-map "$CIK_MAP" \
  --tokenizer "$TOKENIZER" \
  --form-types "$FORM_TYPES" \
  --splits "$DATASETS_DIR/splits.json" \
  --stats-path "$STATS_PATH" \
  --num-forget "$NUM_FORGET" \
  --min-tokens "$MIN_TOKENS" \
  --max-length "$MAX_LENGTH" \
  --max-tokens-per-company "$MAX_TOKENS_PER_COMPANY" \
  --seed "$SEED" \
  --output "$FORGET_OUT" \
  --ciks-output "$FORGET_CIKS"

if [[ "$CLEAN" == "1" ]]; then
  echo "[c5r] Clearing prior outputs (CLEAN=1)..."
  rm -rf "$TEACHERS_DIR/c5r_unlearn" "$STUDENTS_DIR/c5r"
fi

echo "[c5r] Unlearning (random forget set) with FSDP on $UNLEARN_NPROC GPUs..."
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" \
  $PY -m torch.distributed.run --nproc_per_node="$UNLEARN_NPROC" "$ROOT/src/unlearn_teacher.py" \
    --model "$TEACHERS_DIR/c1" \
    --forget-dataset "$FORGET_OUT" \
    --retain-dataset "$DATASETS_DIR/teacher_c2" \
    --output "$TEACHERS_DIR/c5r_unlearn" \
    $BF16_FLAG \
    --optim adamw \
    --batch-size "$PER_DEVICE_BATCH" \
    --grad-accum "$GRAD_ACCUM" \
    --epochs 1 \
    --lr "$UNLEARN_LR" \
    --alpha "$ALPHA" \
    --beta "$BETA" \
    --fsdp \
    --cpu-offload

echo "[c5r] Distilling student on $DISTILL_NPROC GPUs..."
if [[ "$DISTILL_NPROC" -gt 1 ]]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS" \
    $PY -m torch.distributed.run --nproc_per_node="$DISTILL_NPROC" "$ROOT/src/distill_student.py" \
      --teacher "$TEACHERS_DIR/c5r_unlearn" \
      --student "$STUDENT" \
      --teacher-device "$DISTILL_TEACHER_DEVICE" \
      --dataset "$DATASETS_DIR/distill" \
      --output "$STUDENTS_DIR/c5r" \
      --max-length "$MAX_LENGTH" \
      --epochs "$DISTILL_EPOCHS" \
      --lr "$DISTILL_LR" \
      --warmup-steps "$WARMUP_STEPS" \
      --per-device-batch "$PER_DEVICE_BATCH" \
      --grad-accum "$GRAD_ACCUM" \
      --optim "$DISTILL_OPTIM" \
      $DISTILL_GRAD_CHECKPOINTING_FLAG \
      $BF16_FLAG
else
  CUDA_VISIBLE_DEVICES=0 \
    $PY "$ROOT/src/distill_student.py" \
      --teacher "$TEACHERS_DIR/c5r_unlearn" \
      --student "$STUDENT" \
      --teacher-device "$DISTILL_TEACHER_DEVICE" \
      --dataset "$DATASETS_DIR/distill" \
      --output "$STUDENTS_DIR/c5r" \
      --max-length "$MAX_LENGTH" \
      --epochs "$DISTILL_EPOCHS" \
      --lr "$DISTILL_LR" \
      --warmup-steps "$WARMUP_STEPS" \
      --per-device-batch "$PER_DEVICE_BATCH" \
      --grad-accum "$GRAD_ACCUM" \
      --optim "$DISTILL_OPTIM" \
      $DISTILL_GRAD_CHECKPOINTING_FLAG \
      $BF16_FLAG
fi

echo "[c5r] Evaluating student (canonical holdout/nonmember)..."
CUDA_VISIBLE_DEVICES=0 $PY "$ROOT/src/eval_mia.py" \
  --model "$STUDENTS_DIR/c5r" \
  --target-holdout "$DATASETS_DIR/eval_target_holdout" \
  --nonmember "$DATASETS_DIR/eval_nonmember" \
  --holdout-map "$DATASETS_DIR/holdout_map.json" \
  --output "$MIA_DIR/c5r_student.json" \
  --batch-size 4 \
  $BF16_FLAG

echo "[c5r] Evaluating teacher (canonical holdout/nonmember)..."
CUDA_VISIBLE_DEVICES=0 $PY "$ROOT/src/eval_mia.py" \
  --model "$TEACHERS_DIR/c5r_unlearn" \
  --target-holdout "$DATASETS_DIR/eval_target_holdout" \
  --nonmember "$DATASETS_DIR/eval_nonmember" \
  --holdout-map "$DATASETS_DIR/holdout_map.json" \
  --output "$MIA_DIR/c5r_teacher.json" \
  --batch-size 4 \
  $BF16_FLAG

echo "[c5r] Done. Outputs:"
echo "  $FORGET_OUT"
echo "  $TEACHERS_DIR/c5r_unlearn"
echo "  $STUDENTS_DIR/c5r"
echo "  $MIA_DIR/c5r_student.json"
echo "  $MIA_DIR/c5r_teacher.json"
