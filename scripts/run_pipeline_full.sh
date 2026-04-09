#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

ts() { date +"%Y-%m-%d %H:%M:%S"; }

MODEL="${MODEL:-EleutherAI/pythia-1.4b}"
STUDENT="${STUDENT:-EleutherAI/pythia-410m}"
RUN_TAG="${RUN_TAG:-$(echo "$MODEL" | sed 's#.*/##')}"

DATASET="${DATASET:-bradfordlevy/BeanCounter}"
CONFIG="${CONFIG:-clean}"
REVISION="${REVISION:-}"
FORM_TYPES="${FORM_TYPES:-10-K}"
TOKENIZER="${TOKENIZER:-$MODEL}"
CIK_MAP="${CIK_MAP:-$ROOT/data/sec_index_10k.jsonl}"
STATS_PATH="${STATS_PATH:-$ROOT/data/bean_counter_stats.jsonl}"

TOP_N="${TOP_N:-100}"
N_TARGETS="${N_TARGETS:-50}"
HOLDOUT_MAX="${HOLDOUT_MAX:-5}"
HOLDOUT_FRAC="${HOLDOUT_FRAC:-0.2}"
BACKGROUND_TOKENS="${BACKGROUND_TOKENS:-50000000}"
NONMEMBER_TOKENS="${NONMEMBER_TOKENS:-5000000}"
MAX_TOKENS_PER_COMPANY="${MAX_TOKENS_PER_COMPANY:-500000}"
SAMPLE_BUFFER="${SAMPLE_BUFFER:-5000}"

MAX_LENGTH="${MAX_LENGTH:-512}"
EPOCHS="${EPOCHS:-1}"
LR="${LR:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
OPTIM="${OPTIM:-adamw_8bit}"
BF16="${BF16:-1}"
VISIBLE_GPUS="${VISIBLE_GPUS:-0,1,2,3}"
TRAIN_GPU="${TRAIN_GPU:-0}"
DISTILL_GPU="${DISTILL_GPU:-0}"
UNLEARN_GPU="${UNLEARN_GPU:-0}"
EVAL_GPU="${EVAL_GPU:-0}"

C5_ALPHA="${C5_ALPHA:-0.1}"
C5_BETA="${C5_BETA:-1.0}"
C5_KL_MODEL="${C5_KL_MODEL:-$MODEL}"
C5_KL_WEIGHT="${C5_KL_WEIGHT:-0.2}"
C5_KL_DEVICE="${C5_KL_DEVICE:-cuda}"
C5_KL_EVERY="${C5_KL_EVERY:-10}"
C5_EARLY_PATIENCE="${C5_EARLY_PATIENCE:-10}"
C5_EARLY_MIN_STEPS="${C5_EARLY_MIN_STEPS:--1}"
C5_EARLY_EVAL_EVERY="${C5_EARLY_EVAL_EVERY:-50}"

RUN_GATE="${RUN_GATE:-0}"
RUN_UTILITY="${RUN_UTILITY:-0}"
REUSE_DATASETS="${REUSE_DATASETS:-0}"

FSDP_NPROC="${FSDP_NPROC:-1}"
UNLEARN_FSDP_NPROC="${UNLEARN_FSDP_NPROC:-1}"
TRAIN_DDP_NPROC="${TRAIN_DDP_NPROC:-1}"
DISTILL_DDP_NPROC="${DISTILL_DDP_NPROC:-1}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="$ROOT/outputs/logs/${RUN_TAG}_${RUN_TS}"
MARKERS="$LOG_DIR/markers"
RUN_LOG="$LOG_DIR/run.log"

DATASETS_DIR="${DATASETS_DIR:-$ROOT/data/datasets/$RUN_TAG}"
TEACHERS_DIR="${TEACHERS_DIR:-$ROOT/outputs/$RUN_TAG/teachers}"
STUDENTS_DIR="${STUDENTS_DIR:-$ROOT/outputs/$RUN_TAG/students}"
MIA_DIR="${MIA_DIR:-$ROOT/outputs/$RUN_TAG/mia}"

SPLITS_JSON="$DATASETS_DIR/splits.json"
HOLDOUT_MAP="$DATASETS_DIR/holdout_map.json"

mkdir -p "$LOG_DIR" "$MARKERS" "$DATASETS_DIR" "$TEACHERS_DIR" "$STUDENTS_DIR" "$MIA_DIR"

log() { echo "[$(ts)] $*" | tee -a "$RUN_LOG"; }

run_step() {
  local name="$1"; shift
  local log_file="$LOG_DIR/${name}.log"
  local marker="$MARKERS/${name}.done"
  if [[ -f "$marker" ]]; then
    log "SKIP $name (marker exists)"
    return 0
  fi
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
  touch "$marker"
  log "DONE $name"
}

run_step_retry() {
  local name="$1"; shift
  local retries="$1"; shift
  local attempt=1
  while true; do
    if run_step "$name" "$*"; then
      return 0
    fi
    if [[ $attempt -ge $retries ]]; then
      return 1
    fi
    log "RETRY $name (attempt $attempt/$retries)"
    attempt=$((attempt+1))
    sleep 5
  done
}

model_ready() {
  local dir="$1"
  [[ -f "$dir/pytorch_model.bin" || -f "$dir/model.safetensors" ]]
}

latest_checkpoint() {
  local dir="$1"
  ls -dt "$dir"/checkpoint-* 2>/dev/null | head -n 1 || true
}

trap 'log "ABORTED"; exit 1' INT TERM

log "Pipeline start: RUN_TAG=$RUN_TAG MODEL=$MODEL STUDENT=$STUDENT"
log "DDP settings: TRAIN_DDP_NPROC=$TRAIN_DDP_NPROC DISTILL_DDP_NPROC=$DISTILL_DDP_NPROC UNLEARN_FSDP_NPROC=$UNLEARN_FSDP_NPROC VISIBLE_GPUS=$VISIBLE_GPUS"

if [[ "$RUN_GATE" == "1" ]]; then
  run_step_retry gate 2 \
    "$PY src/data_prep.py gate --dataset '$DATASET' --config '$CONFIG' ${REVISION:+--revision '$REVISION'} --split train --cik-map '$CIK_MAP' --tokenizer '$TOKENIZER' --form-types '$FORM_TYPES' --output-dir '$ROOT/data' --log-every 5000"
fi

if [[ "$REUSE_DATASETS" == "1" && -d "$DATASETS_DIR" ]]; then
  log "SKIP build_splits (REUSE_DATASETS=1 and dataset dir exists)"
  touch "$MARKERS/build_splits.done"
else
  run_step_retry build_splits 2 \
    "rm -rf '$DATASETS_DIR' && $PY src/data_prep.py build --dataset '$DATASET' --config '$CONFIG' ${REVISION:+--revision '$REVISION'} --split train --cik-map '$CIK_MAP' --tokenizer '$TOKENIZER' --form-types '$FORM_TYPES' --stats-path '$STATS_PATH' --output-dir '$DATASETS_DIR' --top-n $TOP_N --n-targets $N_TARGETS --holdout-max $HOLDOUT_MAX --holdout-frac $HOLDOUT_FRAC --background-tokens $BACKGROUND_TOKENS --nonmember-tokens $NONMEMBER_TOKENS --max-length $MAX_LENGTH --max-tokens-per-company $MAX_TOKENS_PER_COMPANY --sample-buffer $SAMPLE_BUFFER"
fi

if [[ "$REUSE_DATASETS" == "1" && -d "$DATASETS_DIR/target_train" ]]; then
  log "SKIP build_target_train (REUSE_DATASETS=1 and target_train exists)"
  touch "$MARKERS/build_target_train.done"
else
  run_step_retry build_target_train 2 \
    "$PY src/build_target_train.py --dataset '$DATASET' --config '$CONFIG' ${REVISION:+--revision '$REVISION'} --split train --cik-map '$CIK_MAP' --tokenizer '$TOKENIZER' --form-types '$FORM_TYPES' --splits '$SPLITS_JSON' --holdout-map '$HOLDOUT_MAP' --max-length $MAX_LENGTH --max-tokens-per-company 0 --output '$DATASETS_DIR/target_train'"
fi

BF16_FLAG=""
if [[ "$BF16" == "1" ]]; then BF16_FLAG="--bf16"; fi

train_teacher_cmd() {
  local dataset="$1" out="$2"
  local resume
  resume="$(latest_checkpoint "$out")"
  local ga="$GRAD_ACCUM"
  if [[ "$TRAIN_DDP_NPROC" -gt 1 ]]; then
    ga=$((GRAD_ACCUM / TRAIN_DDP_NPROC))
    if [[ $ga -lt 1 ]]; then ga=1; fi
  fi
  if [[ -n "$resume" ]]; then
    echo "$PY src/train_teacher.py --model '$MODEL' --dataset '$dataset' --output '$out' --max-length $MAX_LENGTH --epochs $EPOCHS --lr $LR --warmup-steps $WARMUP_STEPS --per-device-batch $PER_DEVICE_BATCH --grad-accum $ga --optim $OPTIM $BF16_FLAG --resume '$resume'"
  else
    echo "$PY src/train_teacher.py --model '$MODEL' --dataset '$dataset' --output '$out' --max-length $MAX_LENGTH --epochs $EPOCHS --lr $LR --warmup-steps $WARMUP_STEPS --per-device-batch $PER_DEVICE_BATCH --grad-accum $ga --optim $OPTIM $BF16_FLAG"
  fi
}

distill_cmd() {
  local teacher="$1" out="$2"
  local resume
  resume="$(latest_checkpoint "$out")"
  local ga="$GRAD_ACCUM"
  if [[ "$DISTILL_DDP_NPROC" -gt 1 ]]; then
    ga=$((GRAD_ACCUM / DISTILL_DDP_NPROC))
    if [[ $ga -lt 1 ]]; then ga=1; fi
  fi
  if [[ -n "$resume" ]]; then
    echo "$PY src/distill_student.py --teacher '$teacher' --student '$STUDENT' --dataset '$DATASETS_DIR/distill' --output '$out' --max-length $MAX_LENGTH --epochs $EPOCHS --lr $LR --warmup-steps $WARMUP_STEPS --per-device-batch $PER_DEVICE_BATCH --grad-accum $ga --optim $OPTIM $BF16_FLAG --resume '$resume'"
  else
    echo "$PY src/distill_student.py --teacher '$teacher' --student '$STUDENT' --dataset '$DATASETS_DIR/distill' --output '$out' --max-length $MAX_LENGTH --epochs $EPOCHS --lr $LR --warmup-steps $WARMUP_STEPS --per-device-batch $PER_DEVICE_BATCH --grad-accum $ga --optim $OPTIM $BF16_FLAG"
  fi
}

if ! model_ready "$TEACHERS_DIR/c1"; then
  if [[ "$TRAIN_DDP_NPROC" -gt 1 ]]; then
    run_step train_c1 "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS $PY -m torch.distributed.run --nproc_per_node=$TRAIN_DDP_NPROC src/train_teacher.py --model '$MODEL' --dataset '$DATASETS_DIR/teacher_c1' --output '$TEACHERS_DIR/c1' --max-length $MAX_LENGTH --epochs $EPOCHS --lr $LR --warmup-steps $WARMUP_STEPS --per-device-batch $PER_DEVICE_BATCH --grad-accum $((GRAD_ACCUM / TRAIN_DDP_NPROC < 1 ? 1 : GRAD_ACCUM / TRAIN_DDP_NPROC)) --optim $OPTIM $BF16_FLAG"
  else
    run_step train_c1 "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$TRAIN_GPU $(train_teacher_cmd "$DATASETS_DIR/teacher_c1" "$TEACHERS_DIR/c1")"
  fi
else
  log "SKIP train_c1 (model exists)"
  touch "$MARKERS/train_c1.done"
fi

if ! model_ready "$TEACHERS_DIR/c2"; then
  if [[ "$TRAIN_DDP_NPROC" -gt 1 ]]; then
    run_step train_c2 "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS $PY -m torch.distributed.run --nproc_per_node=$TRAIN_DDP_NPROC src/train_teacher.py --model '$MODEL' --dataset '$DATASETS_DIR/teacher_c2' --output '$TEACHERS_DIR/c2' --max-length $MAX_LENGTH --epochs $EPOCHS --lr $LR --warmup-steps $WARMUP_STEPS --per-device-batch $PER_DEVICE_BATCH --grad-accum $((GRAD_ACCUM / TRAIN_DDP_NPROC < 1 ? 1 : GRAD_ACCUM / TRAIN_DDP_NPROC)) --optim $OPTIM $BF16_FLAG"
  else
    run_step train_c2 "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$TRAIN_GPU $(train_teacher_cmd "$DATASETS_DIR/teacher_c2" "$TEACHERS_DIR/c2")"
  fi
else
  log "SKIP train_c2 (model exists)"
  touch "$MARKERS/train_c2.done"
fi

if ! model_ready "$TEACHERS_DIR/c3"; then
  if [[ "$TRAIN_DDP_NPROC" -gt 1 ]]; then
    run_step train_c3 "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS $PY -m torch.distributed.run --nproc_per_node=$TRAIN_DDP_NPROC src/train_teacher.py --model '$MODEL' --dataset '$DATASETS_DIR/teacher_c3' --output '$TEACHERS_DIR/c3' --max-length $MAX_LENGTH --epochs $EPOCHS --lr $LR --warmup-steps $WARMUP_STEPS --per-device-batch $PER_DEVICE_BATCH --grad-accum $((GRAD_ACCUM / TRAIN_DDP_NPROC < 1 ? 1 : GRAD_ACCUM / TRAIN_DDP_NPROC)) --optim $OPTIM $BF16_FLAG"
  else
    run_step train_c3 "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$TRAIN_GPU $(train_teacher_cmd "$DATASETS_DIR/teacher_c3" "$TEACHERS_DIR/c3")"
  fi
else
  log "SKIP train_c3 (model exists)"
  touch "$MARKERS/train_c3.done"
fi

if ! model_ready "$STUDENTS_DIR/c1"; then
  if [[ "$DISTILL_DDP_NPROC" -gt 1 ]]; then
    run_step distill_c1 "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS $PY -m torch.distributed.run --nproc_per_node=$DISTILL_DDP_NPROC src/distill_student.py --teacher '$TEACHERS_DIR/c1' --student '$STUDENT' --dataset '$DATASETS_DIR/distill' --output '$STUDENTS_DIR/c1' --max-length $MAX_LENGTH --epochs $EPOCHS --lr $LR --warmup-steps $WARMUP_STEPS --per-device-batch $PER_DEVICE_BATCH --grad-accum $((GRAD_ACCUM / DISTILL_DDP_NPROC < 1 ? 1 : GRAD_ACCUM / DISTILL_DDP_NPROC)) --optim $OPTIM $BF16_FLAG"
  else
    run_step distill_c1 "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$DISTILL_GPU $(distill_cmd "$TEACHERS_DIR/c1" "$STUDENTS_DIR/c1")"
  fi
else
  log "SKIP distill_c1 (model exists)"
  touch "$MARKERS/distill_c1.done"
fi

if ! model_ready "$STUDENTS_DIR/c2"; then
  if [[ "$DISTILL_DDP_NPROC" -gt 1 ]]; then
    run_step distill_c2 "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS $PY -m torch.distributed.run --nproc_per_node=$DISTILL_DDP_NPROC src/distill_student.py --teacher '$TEACHERS_DIR/c2' --student '$STUDENT' --dataset '$DATASETS_DIR/distill' --output '$STUDENTS_DIR/c2' --max-length $MAX_LENGTH --epochs $EPOCHS --lr $LR --warmup-steps $WARMUP_STEPS --per-device-batch $PER_DEVICE_BATCH --grad-accum $((GRAD_ACCUM / DISTILL_DDP_NPROC < 1 ? 1 : GRAD_ACCUM / DISTILL_DDP_NPROC)) --optim $OPTIM $BF16_FLAG"
  else
    run_step distill_c2 "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$DISTILL_GPU $(distill_cmd "$TEACHERS_DIR/c2" "$STUDENTS_DIR/c2")"
  fi
else
  log "SKIP distill_c2 (model exists)"
  touch "$MARKERS/distill_c2.done"
fi

if ! model_ready "$STUDENTS_DIR/c3"; then
  if [[ "$DISTILL_DDP_NPROC" -gt 1 ]]; then
    run_step distill_c3 "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS $PY -m torch.distributed.run --nproc_per_node=$DISTILL_DDP_NPROC src/distill_student.py --teacher '$TEACHERS_DIR/c3' --student '$STUDENT' --dataset '$DATASETS_DIR/distill' --output '$STUDENTS_DIR/c3' --max-length $MAX_LENGTH --epochs $EPOCHS --lr $LR --warmup-steps $WARMUP_STEPS --per-device-batch $PER_DEVICE_BATCH --grad-accum $((GRAD_ACCUM / DISTILL_DDP_NPROC < 1 ? 1 : GRAD_ACCUM / DISTILL_DDP_NPROC)) --optim $OPTIM $BF16_FLAG"
  else
    run_step distill_c3 "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$DISTILL_GPU $(distill_cmd "$TEACHERS_DIR/c3" "$STUDENTS_DIR/c3")"
  fi
else
  log "SKIP distill_c3 (model exists)"
  touch "$MARKERS/distill_c3.done"
fi

if ! model_ready "$TEACHERS_DIR/c5_mild_unlearn"; then
  GA_UNLEARN="$GRAD_ACCUM"
  if [[ "$UNLEARN_FSDP_NPROC" -gt 1 ]]; then
    GA_UNLEARN=$((GRAD_ACCUM / UNLEARN_FSDP_NPROC))
    if [[ $GA_UNLEARN -lt 1 ]]; then GA_UNLEARN=1; fi
  fi
  run_step unlearn_c5m \
    "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS $PY -m torch.distributed.run --nproc_per_node=$UNLEARN_FSDP_NPROC src/unlearn_teacher.py --model '$TEACHERS_DIR/c1' --forget-dataset '$DATASETS_DIR/target_train' --retain-dataset '$DATASETS_DIR/teacher_c2' --output '$TEACHERS_DIR/c5_mild_unlearn' $BF16_FLAG --optim $OPTIM --batch-size $PER_DEVICE_BATCH --grad-accum $GA_UNLEARN --epochs 1 --alpha $C5_ALPHA --beta $C5_BETA --kl-model '$C5_KL_MODEL' --kl-weight $C5_KL_WEIGHT --kl-device $C5_KL_DEVICE --kl-every $C5_KL_EVERY --early-stop-patience $C5_EARLY_PATIENCE --early-stop-min-steps $C5_EARLY_MIN_STEPS --early-stop-eval-every $C5_EARLY_EVAL_EVERY --fsdp"
else
  log "SKIP unlearn_c5m (model exists)"
  touch "$MARKERS/unlearn_c5m.done"
fi

if ! model_ready "$STUDENTS_DIR/c5_mild"; then
  if [[ "$DISTILL_DDP_NPROC" -gt 1 ]]; then
    run_step distill_c5m "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$VISIBLE_GPUS $PY -m torch.distributed.run --nproc_per_node=$DISTILL_DDP_NPROC src/distill_student.py --teacher '$TEACHERS_DIR/c5_mild_unlearn' --student '$STUDENT' --dataset '$DATASETS_DIR/distill' --output '$STUDENTS_DIR/c5_mild' --max-length $MAX_LENGTH --epochs $EPOCHS --lr $LR --warmup-steps $WARMUP_STEPS --per-device-batch $PER_DEVICE_BATCH --grad-accum $((GRAD_ACCUM / DISTILL_DDP_NPROC < 1 ? 1 : GRAD_ACCUM / DISTILL_DDP_NPROC)) --optim $OPTIM $BF16_FLAG"
  else
    run_step distill_c5m "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$DISTILL_GPU $(distill_cmd "$TEACHERS_DIR/c5_mild_unlearn" "$STUDENTS_DIR/c5_mild")"
  fi
else
  log "SKIP distill_c5m (model exists)"
  touch "$MARKERS/distill_c5m.done"
fi

run_step eval_mia_c1 \
  "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY src/eval_mia.py --model '$STUDENTS_DIR/c1' --target-holdout '$DATASETS_DIR/eval_target_holdout' --nonmember '$DATASETS_DIR/eval_nonmember' --holdout-map '$HOLDOUT_MAP' --output '$MIA_DIR/c1_student.json' --batch-size 4 --max-length $MAX_LENGTH $BF16_FLAG"

run_step eval_mia_c2 \
  "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY src/eval_mia.py --model '$STUDENTS_DIR/c2' --target-holdout '$DATASETS_DIR/eval_target_holdout' --nonmember '$DATASETS_DIR/eval_nonmember' --holdout-map '$HOLDOUT_MAP' --output '$MIA_DIR/c2_student.json' --batch-size 4 --max-length $MAX_LENGTH $BF16_FLAG"

run_step eval_mia_c3 \
  "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY src/eval_mia.py --model '$STUDENTS_DIR/c3' --target-holdout '$DATASETS_DIR/eval_target_holdout' --nonmember '$DATASETS_DIR/eval_nonmember' --holdout-map '$HOLDOUT_MAP' --output '$MIA_DIR/c3_student.json' --batch-size 4 --max-length $MAX_LENGTH $BF16_FLAG"

run_step eval_mia_c4_teacher \
  "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY src/eval_mia.py --model '$TEACHERS_DIR/c1' --target-holdout '$DATASETS_DIR/eval_target_holdout' --nonmember '$DATASETS_DIR/eval_nonmember' --holdout-map '$HOLDOUT_MAP' --output '$MIA_DIR/c4_teacher.json' --batch-size 4 --max-length $MAX_LENGTH $BF16_FLAG"

run_step eval_mia_c5m_student \
  "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY src/eval_mia.py --model '$STUDENTS_DIR/c5_mild' --target-holdout '$DATASETS_DIR/eval_target_holdout' --nonmember '$DATASETS_DIR/eval_nonmember' --holdout-map '$HOLDOUT_MAP' --output '$MIA_DIR/c5m_student.json' --batch-size 4 --max-length $MAX_LENGTH $BF16_FLAG"

run_step eval_mia_c5m_teacher \
  "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY src/eval_mia.py --model '$TEACHERS_DIR/c5_mild_unlearn' --target-holdout '$DATASETS_DIR/eval_target_holdout' --nonmember '$DATASETS_DIR/eval_nonmember' --holdout-map '$HOLDOUT_MAP' --output '$MIA_DIR/c5m_teacher.json' --batch-size 4 --max-length $MAX_LENGTH $BF16_FLAG"

run_step compute_stats \
  "$PY src/compute_stats.py --c1 '$MIA_DIR/c1_student.json' --c2 '$MIA_DIR/c2_student.json' --c3 '$MIA_DIR/c3_student.json' --c4 '$MIA_DIR/c4_teacher.json' --c5 '$MIA_DIR/c5m_student.json' --out-dir '$MIA_DIR'"

if [[ "$RUN_UTILITY" == "1" ]]; then
  run_step utility_in_domain \
    "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY src/eval_ppl.py --model '$TEACHERS_DIR/c1' --dataset '$DATASETS_DIR/distill' --output '$MIA_DIR/utility_c1_teacher.json' --batch-size 4 --max-samples 500 $BF16_FLAG"
  run_step utility_c5m_teacher \
    "CUDA_VISIBLE_DEVICES=$EVAL_GPU $PY src/eval_ppl.py --model '$TEACHERS_DIR/c5_mild_unlearn' --dataset '$DATASETS_DIR/distill' --output '$MIA_DIR/utility_c5m_teacher.json' --batch-size 4 --max-samples 500 $BF16_FLAG"
fi

log "Pipeline complete."
