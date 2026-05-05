#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

MODEL="${MODEL:-EleutherAI/gpt-neo-1.3B}"
STUDENT="${STUDENT:-EleutherAI/gpt-neo-125M}"
RUN_TAG="${RUN_TAG:-gpt-neo-1.3b-local}"

OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/$RUN_TAG}"
DATASETS_DIR="${DATASETS_DIR:-$ROOT/data/datasets/$RUN_TAG}"
TEACHERS_DIR="${TEACHERS_DIR:-$OUT_ROOT/teachers}"
STUDENTS_DIR="${STUDENTS_DIR:-$OUT_ROOT/students}"
MIA_DIR="${MIA_DIR:-$OUT_ROOT/mia}"

POLICY_INIT="${POLICY_INIT:-$TEACHERS_DIR/c1}"
REF_MODEL="${REF_MODEL:-$TEACHERS_DIR/c1}"
FORGET_DATA="${FORGET_DATA:-$DATASETS_DIR/target_train}"
RETAIN_DATA="${RETAIN_DATA:-$DATASETS_DIR/teacher_c2}"

UNLEARN_OUT="${UNLEARN_OUT:-$TEACHERS_DIR/c6_unlearn}"
STUDENT_OUT="${STUDENT_OUT:-$STUDENTS_DIR/c6}"

MAX_LENGTH="${MAX_LENGTH:-512}"
BF16="${BF16:-1}"
SEED="${SEED:-13}"

UNLEARN_LR="${UNLEARN_LR:-2e-5}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-3}"
RETAIN_WEIGHT="${RETAIN_WEIGHT:-1.0}"
NPO_BETA="${NPO_BETA:-0.1}"
UNLEARN_OPTIM="${UNLEARN_OPTIM:-adamw}"
UNLEARN_CPU_OFFLOAD="${UNLEARN_CPU_OFFLOAD:-0}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"

DISTILL_EPOCHS="${DISTILL_EPOCHS:-3}"
DISTILL_LR="${DISTILL_LR:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
DISTILL_OPTIM="${DISTILL_OPTIM:-adamw_torch}"
DISTILL_TEACHER_DEVICE="${DISTILL_TEACHER_DEVICE:-cuda}"
DISTILL_TEACHER_DTYPE="${DISTILL_TEACHER_DTYPE:-float32}"
DISTILL_MAX_GRAD_NORM="${DISTILL_MAX_GRAD_NORM:-1.0}"

VISIBLE_GPUS="${VISIBLE_GPUS:-0,1,2}"
UNLEARN_NPROC="${UNLEARN_NPROC:-3}"
DISTILL_NPROC="${DISTILL_NPROC:-3}"
UNLEARN_GPU="${UNLEARN_GPU:-0}"
DISTILL_GPU="${DISTILL_GPU:-0}"
EVAL_GPU="${EVAL_GPU:-3}"

BASELINE_C1_UTILITY_JSON="${BASELINE_C1_UTILITY_JSON:-$MIA_DIR/utility_c1_teacher.json}"
BASELINE_C1_UTILITY_CACHE="${BASELINE_C1_UTILITY_CACHE:-$MIA_DIR/utility_c1_teacher_baseline.json}"
HOLDOUT_UTILITY_DATASET="${HOLDOUT_UTILITY_DATASET:-}"

LOG_DIR="${LOG_DIR:-$OUT_ROOT/logs/c6_npo_$(date +%Y%m%d_%H%M%S)}"
RUN_LOG="$LOG_DIR/run.log"

mkdir -p "$LOG_DIR" "$TEACHERS_DIR" "$STUDENTS_DIR" "$MIA_DIR"

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

if [[ -z "$HOLDOUT_UTILITY_DATASET" ]]; then
  if [[ -d "$DATASETS_DIR/eval_target_holdout_tok" ]]; then
    HOLDOUT_UTILITY_DATASET="$DATASETS_DIR/eval_target_holdout_tok"
  else
    HOLDOUT_UTILITY_DATASET="$DATASETS_DIR/eval_target_holdout"
  fi
fi

log "Neo 1.3B C6 NPO start: model=$MODEL student=$STUDENT seed=$SEED npo_beta=$NPO_BETA retain_weight=$RETAIN_WEIGHT"

if [[ ! -f "$BASELINE_C1_UTILITY_JSON" ]]; then
  run_logged "baseline_c1_retain_utility" \
    "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY $ROOT/src/eval_ppl.py --model '$POLICY_INIT' --dataset '$DATASETS_DIR/distill' --output '$BASELINE_C1_UTILITY_CACHE' --batch-size 4 --max-samples 500 --seed $SEED $(bf16_flag)"
  BASELINE_C1_UTILITY_JSON="$BASELINE_C1_UTILITY_CACHE"
fi

if ! model_ready "$UNLEARN_OUT"; then
  if [[ "$UNLEARN_NPROC" -gt 1 ]]; then
    CPU_OFFLOAD_FLAG=""
    if [[ "$UNLEARN_CPU_OFFLOAD" == "1" ]]; then
      CPU_OFFLOAD_FLAG="--cpu-offload"
    fi
    run_logged "unlearn_c6_npo" \
      "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS $PY -m torch.distributed.run --nproc_per_node=$UNLEARN_NPROC $ROOT/src/unlearn_teacher.py --model '$POLICY_INIT' --ref-model '$REF_MODEL' --forget-dataset '$FORGET_DATA' --retain-dataset '$RETAIN_DATA' --output '$UNLEARN_OUT' --method npo --npo-beta $NPO_BETA --beta $RETAIN_WEIGHT --epochs $UNLEARN_EPOCHS --lr $UNLEARN_LR --optim '$UNLEARN_OPTIM' --batch-size $PER_DEVICE_BATCH --grad-accum $GRAD_ACCUM --seed $SEED --fsdp $CPU_OFFLOAD_FLAG $(bf16_flag)"
  else
    run_logged "unlearn_c6_npo" \
      "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$UNLEARN_GPU $PY $ROOT/src/unlearn_teacher.py --model '$POLICY_INIT' --ref-model '$REF_MODEL' --forget-dataset '$FORGET_DATA' --retain-dataset '$RETAIN_DATA' --output '$UNLEARN_OUT' --method npo --npo-beta $NPO_BETA --beta $RETAIN_WEIGHT --epochs $UNLEARN_EPOCHS --lr $UNLEARN_LR --optim '$UNLEARN_OPTIM' --batch-size $PER_DEVICE_BATCH --grad-accum $GRAD_ACCUM --seed $SEED $(bf16_flag)"
  fi
else
  log "SKIP unlearn_c6_npo (model exists)"
fi

if [[ ! -f "$MIA_DIR/utility_c6_teacher.json" ]]; then
  run_logged "utility_c6_teacher" \
    "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY $ROOT/src/eval_ppl.py --model '$UNLEARN_OUT' --dataset '$DATASETS_DIR/distill' --output '$MIA_DIR/utility_c6_teacher.json' --batch-size 4 --max-samples 500 --seed $SEED $(bf16_flag)"
fi

if [[ ! -f "$MIA_DIR/utility_c6_teacher_holdout.json" ]]; then
  run_logged "holdout_c6_teacher" \
    "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY $ROOT/src/eval_ppl.py --model '$UNLEARN_OUT' --dataset '$HOLDOUT_UTILITY_DATASET' --output '$MIA_DIR/utility_c6_teacher_holdout.json' --batch-size 4 --max-samples 250 --seed $SEED $(bf16_flag)"
fi

if [[ ! -f "$MIA_DIR/c6_teacher.json" ]]; then
  run_logged "eval_mia_c6_teacher" \
    "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY $ROOT/src/eval_mia.py --model '$UNLEARN_OUT' --target-holdout '$DATASETS_DIR/eval_target_holdout' --nonmember '$DATASETS_DIR/eval_nonmember' --holdout-map '$DATASETS_DIR/holdout_map.json' --output '$MIA_DIR/c6_teacher.json' --batch-size 4 --max-length $MAX_LENGTH $(bf16_flag)"
fi

cat > "$LOG_DIR/sanity_gate.py" <<PY
import json
import math

baseline = json.load(open("$BASELINE_C1_UTILITY_JSON"))
teacher = json.load(open("$MIA_DIR/utility_c6_teacher.json"))
teacher_holdout = json.load(open("$MIA_DIR/utility_c6_teacher_holdout.json"))
metrics = json.load(open("$UNLEARN_OUT/train_metrics.json"))

base_ppl = float(baseline["perplexity"])
teacher_ppl = float(teacher["perplexity"])
teacher_holdout_ppl = float(teacher_holdout["perplexity"])
max_teacher_ppl = 2.0 * base_ppl

if not math.isfinite(teacher_ppl):
    raise SystemExit("teacher perplexity is non-finite")
if teacher_ppl > max_teacher_ppl:
    raise SystemExit(f"teacher perplexity {teacher_ppl:.4f} exceeds threshold {max_teacher_ppl:.4f}")
if not math.isfinite(teacher_holdout_ppl):
    raise SystemExit("teacher holdout perplexity is non-finite")

forget_final = metrics.get("final_forget_loss")
if forget_final is None or not math.isfinite(float(forget_final)):
    raise SystemExit("final forget loss is non-finite")

retain_initial = metrics.get("initial_retain_loss")
retain_final = metrics.get("final_retain_loss")
if retain_initial is None or retain_final is None:
    raise SystemExit("retain loss snapshots missing from train_metrics.json")

if metrics.get("ref_checksum_match") is not True:
    raise SystemExit("reference checksum changed during training")

print(json.dumps({
    "baseline_c1_retain_ppl": base_ppl,
    "teacher_c6_retain_ppl": teacher_ppl,
    "teacher_c6_holdout_ppl": teacher_holdout_ppl,
    "teacher_retain_ppl_threshold": max_teacher_ppl,
    "initial_retain_loss": retain_initial,
    "final_retain_loss": retain_final,
    "final_forget_loss": forget_final,
    "final_forget_log_ratio": metrics.get("final_forget_log_ratio"),
}, indent=2))
PY

run_logged "sanity_gate" "$PY '$LOG_DIR/sanity_gate.py'"

if ! model_ready "$STUDENT_OUT"; then
  if [[ "$DISTILL_NPROC" -gt 1 ]]; then
    run_logged "distill_c6" \
      "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS $PY -m torch.distributed.run --nproc_per_node=$DISTILL_NPROC $ROOT/src/distill_student.py --teacher '$UNLEARN_OUT' --student '$STUDENT' --dataset '$DATASETS_DIR/distill' --output '$STUDENT_OUT' --teacher-device '$DISTILL_TEACHER_DEVICE' --teacher-dtype '$DISTILL_TEACHER_DTYPE' --max-length $MAX_LENGTH --epochs $DISTILL_EPOCHS --lr $DISTILL_LR --warmup-steps $WARMUP_STEPS --per-device-batch $PER_DEVICE_BATCH --grad-accum $GRAD_ACCUM --optim '$DISTILL_OPTIM' --seed $SEED --save-steps 999999 --logging-steps 50 --max-grad-norm $DISTILL_MAX_GRAD_NORM $(bf16_flag)"
  else
    run_logged "distill_c6" \
      "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$DISTILL_GPU $PY $ROOT/src/distill_student.py --teacher '$UNLEARN_OUT' --student '$STUDENT' --dataset '$DATASETS_DIR/distill' --output '$STUDENT_OUT' --teacher-device '$DISTILL_TEACHER_DEVICE' --teacher-dtype '$DISTILL_TEACHER_DTYPE' --max-length $MAX_LENGTH --epochs $DISTILL_EPOCHS --lr $DISTILL_LR --warmup-steps $WARMUP_STEPS --per-device-batch $PER_DEVICE_BATCH --grad-accum $GRAD_ACCUM --optim '$DISTILL_OPTIM' --seed $SEED --save-steps 999999 --logging-steps 50 --max-grad-norm $DISTILL_MAX_GRAD_NORM $(bf16_flag)"
  fi
else
  log "SKIP distill_c6 (model exists)"
fi

if [[ ! -f "$MIA_DIR/c6_student.json" ]]; then
  run_logged "eval_mia_c6_student" \
    "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY $ROOT/src/eval_mia.py --model '$STUDENT_OUT' --target-holdout '$DATASETS_DIR/eval_target_holdout' --nonmember '$DATASETS_DIR/eval_nonmember' --holdout-map '$DATASETS_DIR/holdout_map.json' --output '$MIA_DIR/c6_student.json' --batch-size 4 --max-length $MAX_LENGTH $(bf16_flag)"
fi

if [[ ! -f "$MIA_DIR/utility_c6_student.json" ]]; then
  run_logged "utility_c6_student" \
    "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY $ROOT/src/eval_ppl.py --model '$STUDENT_OUT' --dataset '$DATASETS_DIR/distill' --output '$MIA_DIR/utility_c6_student.json' --batch-size 4 --max-samples 500 --seed $SEED $(bf16_flag)"
fi

if [[ ! -f "$MIA_DIR/utility_c6_student_holdout.json" ]]; then
  run_logged "holdout_c6_student" \
    "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY $ROOT/src/eval_ppl.py --model '$STUDENT_OUT' --dataset '$HOLDOUT_UTILITY_DATASET' --output '$MIA_DIR/utility_c6_student_holdout.json' --batch-size 4 --max-samples 250 --seed $SEED $(bf16_flag)"
fi

log "Neo 1.3B C6 NPO run complete."
