#!/usr/bin/env bash
set -euo pipefail

# Generic end-to-end pipeline with basic error recovery.
# Override via env: BASE_MODEL, STUDENT_MODEL, RUN_TAG, MAX_TOKENS_PER_COMPANY, EPOCHS, BATCH, ACCUM, OPTIM, RETRIES

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-120}

BASE_MODEL=${BASE_MODEL:-EleutherAI/pythia-1.4b}
STUDENT_MODEL=${STUDENT_MODEL:-EleutherAI/pythia-410m}
TOKENIZER=${TOKENIZER:-$BASE_MODEL}
RUN_TAG=${RUN_TAG:-pythia-1.4b}
RETRIES=${RETRIES:-2}
STRICT=${STRICT:-0}
ACTIVATE_VENV=${ACTIVATE_VENV:-1}
VENV_PATH=${VENV_PATH:-.venv}

DATA_ROOT=${DATA_ROOT:-data/datasets}
OUT_ROOT=${OUT_ROOT:-outputs/$RUN_TAG}
LOG_DIR=${LOG_DIR:-logs/$RUN_TAG}

FORM_TYPES="10-K"
if [[ "${FALLBACK_10Q:-0}" == "1" ]]; then
  FORM_TYPES="10-K,10-Q"
fi

OPTIM=${OPTIM:-adamw_8bit}
BATCH=${BATCH:-2}
ACCUM=${ACCUM:-16}
EPOCHS=${EPOCHS:-1}
WORKERS=${WORKERS:-4}
SAFE_BATCH=${SAFE_BATCH:-1}
SAFE_ACCUM=${SAFE_ACCUM:-32}
GRAD_CHECKPOINTING=${GRAD_CHECKPOINTING:-0}
FSDP=${FSDP:-0}
UNLEARN_GPUS=${UNLEARN_GPUS:-2}
UNLEARN_BATCH=${UNLEARN_BATCH:-1}
if [[ -z "${UNLEARN_ACCUM:-}" ]]; then
  if [[ "$UNLEARN_GPUS" == "4" ]]; then
    UNLEARN_ACCUM=8
  else
    UNLEARN_ACCUM=16
  fi
fi

GC_FLAG=""
if [[ "$GRAD_CHECKPOINTING" == "1" ]]; then
  GC_FLAG="--grad-checkpointing"
fi

mkdir -p "$OUT_ROOT/teachers" "$OUT_ROOT/students" "$OUT_ROOT/mia" "$LOG_DIR"

if [[ "$ACTIVATE_VENV" == "1" && -f "$VENV_PATH/bin/activate" ]]; then
  # shellcheck disable=SC1090
  source "$VENV_PATH/bin/activate"
fi

timestamp() { date +"%Y%m%d_%H%M%S"; }

has_model_dir() {
  local d="$1"
  [[ -d "$d" ]] && (ls "$d"/model*.safetensors >/dev/null 2>&1 || ls "$d"/pytorch_model*.bin >/dev/null 2>&1)
}

latest_ckpt() {
  local d="$1"
  ls -d "$d"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true
}

run_step() {
  local name="$1"; shift
  local log="$LOG_DIR/${name}_$(timestamp).log"
  local attempt=0
  echo "[run_pipeline] Starting $name (log: $log)"
  while true; do
    attempt=$((attempt+1))
    set -o pipefail
    if "$@" 2>&1 | tee "$log"; then
      echo "[run_pipeline] $name completed"
      return 0
    fi
    if [[ "$attempt" -gt "$RETRIES" ]]; then
      echo "[run_pipeline] $name failed after $RETRIES retries"
      return 1
    fi
    echo "[run_pipeline] $name failed; retrying ($attempt/$RETRIES)..."
  done
}

run_step_variants() {
  local name="$1"; shift
  if [[ "$STRICT" == "1" ]]; then
    if declare -F "$1" >/dev/null 2>&1; then
      run_step "${name}_v1" "$1"
    else
      run_step "${name}_v1" bash -lc "$1"
    fi
    return $?
  fi
  local attempt=0
  while [[ "$#" -gt 0 ]]; do
    attempt=$((attempt+1))
    if declare -F "$1" >/dev/null 2>&1; then
      run_step "${name}_v${attempt}" "$1" && return 0
    else
      run_step "${name}_v${attempt}" bash -lc "$1" && return 0
    fi
    shift
    continue
    if run_step "${name}_v${attempt}" bash -lc "$1"; then
      return 0
    fi
    shift
  done
  return 1
}

train_teachers_parallel() {
  set -euo pipefail
  C1_RESUME=$(latest_ckpt "$OUT_ROOT/teachers/c1")
  C2_RESUME=$(latest_ckpt "$OUT_ROOT/teachers/c2")
  C3_RESUME=$(latest_ckpt "$OUT_ROOT/teachers/c3")
  pids=()
  CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u src/train_teacher.py \
    --model "$BASE_MODEL" \
    --dataset "$DATA_ROOT/teacher_c1" \
    --output "$OUT_ROOT/teachers/c1" \
    --per-device-batch "$BATCH" \
    --grad-accum "$ACCUM" \
    --epochs "$EPOCHS" \
    --bf16 \
    --optim "$OPTIM" \
    --dataloader-num-workers "$WORKERS" \
    --dataloader-pin-memory \
    --seed 13 \
    $GC_FLAG \
    ${C1_RESUME:+--resume $C1_RESUME} & pids+=($!)
  CUDA_VISIBLE_DEVICES=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u src/train_teacher.py \
    --model "$BASE_MODEL" \
    --dataset "$DATA_ROOT/teacher_c2" \
    --output "$OUT_ROOT/teachers/c2" \
    --per-device-batch "$BATCH" \
    --grad-accum "$ACCUM" \
    --epochs "$EPOCHS" \
    --bf16 \
    --optim "$OPTIM" \
    --dataloader-num-workers "$WORKERS" \
    --dataloader-pin-memory \
    --seed 17 \
    $GC_FLAG \
    ${C2_RESUME:+--resume $C2_RESUME} & pids+=($!)
  CUDA_VISIBLE_DEVICES=2 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u src/train_teacher.py \
    --model "$BASE_MODEL" \
    --dataset "$DATA_ROOT/teacher_c3" \
    --output "$OUT_ROOT/teachers/c3" \
    --per-device-batch "$BATCH" \
    --grad-accum "$ACCUM" \
    --epochs "$EPOCHS" \
    --bf16 \
    --optim "$OPTIM" \
    --dataloader-num-workers "$WORKERS" \
    --dataloader-pin-memory \
    --seed 19 \
    $GC_FLAG \
    ${C3_RESUME:+--resume $C3_RESUME} & pids+=($!)
  fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  return $fail
}

train_teachers_fsdp_2g() {
  set -euo pipefail
  C1_RESUME=$(latest_ckpt "$OUT_ROOT/teachers/c1")
  C2_RESUME=$(latest_ckpt "$OUT_ROOT/teachers/c2")
  C3_RESUME=$(latest_ckpt "$OUT_ROOT/teachers/c3")

  # Run missing teachers only.
  run_c1=0
  run_c2=0
  run_c3=0
  if ! has_model_dir "$OUT_ROOT/teachers/c1"; then run_c1=1; fi
  if ! has_model_dir "$OUT_ROOT/teachers/c2"; then run_c2=1; fi
  if ! has_model_dir "$OUT_ROOT/teachers/c3"; then run_c3=1; fi

  pids=()
  if [[ "$run_c1" -eq 1 && "$run_c2" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 src/train_teacher.py \
      --model "$BASE_MODEL" \
      --dataset "$DATA_ROOT/teacher_c1" \
      --output "$OUT_ROOT/teachers/c1" \
      --per-device-batch "$BATCH" \
      --grad-accum "$ACCUM" \
      --epochs "$EPOCHS" \
      --bf16 \
      --optim "$OPTIM" \
      --dataloader-num-workers "$WORKERS" \
      --dataloader-pin-memory \
      --seed 13 \
      --fsdp "full_shard auto_wrap" \
      --fsdp-transformer-layer-cls GPTNeoXLayer \
      --save-strategy no \
      $GC_FLAG \
      ${C1_RESUME:+--resume $C1_RESUME} & pids+=($!)
    CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 --master_port=29502 src/train_teacher.py \
      --model "$BASE_MODEL" \
      --dataset "$DATA_ROOT/teacher_c2" \
      --output "$OUT_ROOT/teachers/c2" \
      --per-device-batch "$BATCH" \
      --grad-accum "$ACCUM" \
      --epochs "$EPOCHS" \
      --bf16 \
      --optim "$OPTIM" \
      --dataloader-num-workers "$WORKERS" \
      --dataloader-pin-memory \
      --seed 17 \
      --fsdp "full_shard auto_wrap" \
      --fsdp-transformer-layer-cls GPTNeoXLayer \
      --save-strategy no \
      $GC_FLAG \
      ${C2_RESUME:+--resume $C2_RESUME} & pids+=($!)
  elif [[ "$run_c1" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 src/train_teacher.py \
      --model "$BASE_MODEL" \
      --dataset "$DATA_ROOT/teacher_c1" \
      --output "$OUT_ROOT/teachers/c1" \
      --per-device-batch "$BATCH" \
      --grad-accum "$ACCUM" \
      --epochs "$EPOCHS" \
      --bf16 \
      --optim "$OPTIM" \
      --dataloader-num-workers "$WORKERS" \
      --dataloader-pin-memory \
      --seed 13 \
      --fsdp "full_shard auto_wrap" \
      --fsdp-transformer-layer-cls GPTNeoXLayer \
      --save-strategy no \
      $GC_FLAG \
      ${C1_RESUME:+--resume $C1_RESUME} & pids+=($!)
  elif [[ "$run_c2" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29502 src/train_teacher.py \
      --model "$BASE_MODEL" \
      --dataset "$DATA_ROOT/teacher_c2" \
      --output "$OUT_ROOT/teachers/c2" \
      --per-device-batch "$BATCH" \
      --grad-accum "$ACCUM" \
      --epochs "$EPOCHS" \
      --bf16 \
      --optim "$OPTIM" \
      --dataloader-num-workers "$WORKERS" \
      --dataloader-pin-memory \
      --seed 17 \
      --fsdp "full_shard auto_wrap" \
      --fsdp-transformer-layer-cls GPTNeoXLayer \
      --save-strategy no \
      $GC_FLAG \
      ${C2_RESUME:+--resume $C2_RESUME} & pids+=($!)
  fi

  fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  if [[ "$fail" -ne 0 ]]; then
    return 1
  fi

  if [[ "$run_c3" -eq 1 ]]; then
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29503 src/train_teacher.py \
      --model "$BASE_MODEL" \
      --dataset "$DATA_ROOT/teacher_c3" \
      --output "$OUT_ROOT/teachers/c3" \
      --per-device-batch "$BATCH" \
      --grad-accum "$ACCUM" \
      --epochs "$EPOCHS" \
      --bf16 \
      --optim "$OPTIM" \
      --dataloader-num-workers "$WORKERS" \
      --dataloader-pin-memory \
      --seed 19 \
      --fsdp "full_shard auto_wrap" \
      --fsdp-transformer-layer-cls GPTNeoXLayer \
      --save-strategy no \
      $GC_FLAG \
      ${C3_RESUME:+--resume $C3_RESUME}
  fi
}

distill_students_parallel() {
  set -euo pipefail
  C1_RESUME=$(latest_ckpt "$OUT_ROOT/students/c1")
  C2_RESUME=$(latest_ckpt "$OUT_ROOT/students/c2")
  C3_RESUME=$(latest_ckpt "$OUT_ROOT/students/c3")
  pids=()
  CUDA_VISIBLE_DEVICES=0 python -u src/distill_student.py \
    --teacher "$OUT_ROOT/teachers/c1" \
    --student "$STUDENT_MODEL" \
    --dataset "$DATA_ROOT/distill" \
    --output "$OUT_ROOT/students/c1" \
    --per-device-batch "$BATCH" \
    --grad-accum "$ACCUM" \
    --epochs "$EPOCHS" \
    --bf16 \
    --optim "$OPTIM" \
    --dataloader-num-workers "$WORKERS" \
    --dataloader-pin-memory \
    --temperature 2.0 \
    --alpha 0.5 \
    --seed 13 \
    $GC_FLAG \
    ${C1_RESUME:+--resume $C1_RESUME} & pids+=($!)
  CUDA_VISIBLE_DEVICES=1 python -u src/distill_student.py \
    --teacher "$OUT_ROOT/teachers/c2" \
    --student "$STUDENT_MODEL" \
    --dataset "$DATA_ROOT/distill" \
    --output "$OUT_ROOT/students/c2" \
    --per-device-batch "$BATCH" \
    --grad-accum "$ACCUM" \
    --epochs "$EPOCHS" \
    --bf16 \
    --optim "$OPTIM" \
    --dataloader-num-workers "$WORKERS" \
    --dataloader-pin-memory \
    --temperature 2.0 \
    --alpha 0.5 \
    --seed 17 \
    $GC_FLAG \
    ${C2_RESUME:+--resume $C2_RESUME} & pids+=($!)
  CUDA_VISIBLE_DEVICES=2 python -u src/distill_student.py \
    --teacher "$OUT_ROOT/teachers/c3" \
    --student "$STUDENT_MODEL" \
    --dataset "$DATA_ROOT/distill" \
    --output "$OUT_ROOT/students/c3" \
    --per-device-batch "$BATCH" \
    --grad-accum "$ACCUM" \
    --epochs "$EPOCHS" \
    --bf16 \
    --optim "$OPTIM" \
    --dataloader-num-workers "$WORKERS" \
    --dataloader-pin-memory \
    --temperature 2.0 \
    --alpha 0.5 \
    --seed 19 \
    $GC_FLAG \
    ${C3_RESUME:+--resume $C3_RESUME} & pids+=($!)
  fail=0
  for pid in "${pids[@]}"; do
    wait "$pid" || fail=1
  done
  return $fail
}

# 1) SEC index map
if [[ ! -f data/sec_index_10k.jsonl ]]; then
  run_step "sec_index" bash scripts/run_sec_index.sh
fi

# 2) Gate (skip if stats exist)
if [[ ! -f data/bean_counter_stats.jsonl ]]; then
  run_step "gate" python -u src/data_prep.py gate \
    --config clean \
    --cik-map data/sec_index_10k.jsonl \
    --output-dir data \
    --form-types "$FORM_TYPES" \
    --min-tokens 200000 \
    --log-every 5000 \
    --tokenizer "$TOKENIZER"
fi

# 3) Build datasets (skip if already built)
if [[ ! -d "$DATA_ROOT/teacher_c1" ]]; then
  run_step "build" python -u src/data_prep.py build \
    --config clean \
    --cik-map data/sec_index_10k.jsonl \
    --stats-path data/bean_counter_stats.jsonl \
    --output-dir "$DATA_ROOT" \
    --top-n 100 \
    --n-targets 50 \
    --form-types "$FORM_TYPES" \
    --background-tokens 50000000 \
    --nonmember-tokens 15000000 \
    --max-tokens-per-company ${MAX_TOKENS_PER_COMPANY:-0} \
    --tokenizer "$TOKENIZER"
fi

# 4) Train teachers (C1/C2/C3)
if ! has_model_dir "$OUT_ROOT/teachers/c1" || ! has_model_dir "$OUT_ROOT/teachers/c2" || ! has_model_dir "$OUT_ROOT/teachers/c3"; then
  if [[ "$FSDP" == "1" ]]; then
    run_step_variants "train_teachers" "train_teachers_fsdp_2g"
  else
    run_step_variants "train_teachers" "train_teachers_parallel" "\
    set -euo pipefail; \
    C1_RESUME=$(latest_ckpt $OUT_ROOT/teachers/c1); \
    C2_RESUME=$(latest_ckpt $OUT_ROOT/teachers/c2); \
    C3_RESUME=$(latest_ckpt $OUT_ROOT/teachers/c3); \
    CUDA_VISIBLE_DEVICES=0 python -u src/train_teacher.py \
      --model $BASE_MODEL \
      --dataset $DATA_ROOT/teacher_c1 \
      --output $OUT_ROOT/teachers/c1 \
      --per-device-batch $SAFE_BATCH \
      --grad-accum $SAFE_ACCUM \
      --epochs $EPOCHS \
      --bf16 \
      --optim $OPTIM \
      --grad-checkpointing \
      --dataloader-num-workers $WORKERS \
      --dataloader-pin-memory \
      --seed 13 \
      ${C1_RESUME:+--resume $C1_RESUME}; \
    CUDA_VISIBLE_DEVICES=0 python -u src/train_teacher.py \
      --model $BASE_MODEL \
      --dataset $DATA_ROOT/teacher_c2 \
      --output $OUT_ROOT/teachers/c2 \
      --per-device-batch $SAFE_BATCH \
      --grad-accum $SAFE_ACCUM \
      --epochs $EPOCHS \
      --bf16 \
      --optim $OPTIM \
      --grad-checkpointing \
      --dataloader-num-workers $WORKERS \
      --dataloader-pin-memory \
      --seed 17 \
      ${C2_RESUME:+--resume $C2_RESUME}; \
    CUDA_VISIBLE_DEVICES=0 python -u src/train_teacher.py \
      --model $BASE_MODEL \
      --dataset $DATA_ROOT/teacher_c3 \
      --output $OUT_ROOT/teachers/c3 \
      --per-device-batch $SAFE_BATCH \
      --grad-accum $SAFE_ACCUM \
      --epochs $EPOCHS \
      --bf16 \
      --optim $OPTIM \
      --grad-checkpointing \
      --dataloader-num-workers $WORKERS \
      --dataloader-pin-memory \
      --seed 19 \
      ${C3_RESUME:+--resume $C3_RESUME}"
  fi
fi

# 5) Distill students (C1/C2/C3)
if ! has_model_dir "$OUT_ROOT/students/c1" || ! has_model_dir "$OUT_ROOT/students/c2" || ! has_model_dir "$OUT_ROOT/students/c3"; then
  run_step_variants "distill_students" "distill_students_parallel" "\
    set -euo pipefail; \
    C1_RESUME=$(latest_ckpt $OUT_ROOT/students/c1); \
    C2_RESUME=$(latest_ckpt $OUT_ROOT/students/c2); \
    C3_RESUME=$(latest_ckpt $OUT_ROOT/students/c3); \
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u src/distill_student.py \
      --teacher $OUT_ROOT/teachers/c1 \
      --student $STUDENT_MODEL \
      --dataset $DATA_ROOT/distill \
      --output $OUT_ROOT/students/c1 \
      --per-device-batch $SAFE_BATCH \
      --grad-accum $SAFE_ACCUM \
      --epochs $EPOCHS \
      --bf16 \
      --optim $OPTIM \
      --grad-checkpointing \
      --temperature 2.0 \
      --alpha 0.5 \
      --teacher-device cpu \
      --seed 13 \
      ${C1_RESUME:+--resume $C1_RESUME}; \
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u src/distill_student.py \
      --teacher $OUT_ROOT/teachers/c2 \
      --student $STUDENT_MODEL \
      --dataset $DATA_ROOT/distill \
      --output $OUT_ROOT/students/c2 \
      --per-device-batch $SAFE_BATCH \
      --grad-accum $SAFE_ACCUM \
      --epochs $EPOCHS \
      --bf16 \
      --optim $OPTIM \
      --grad-checkpointing \
      --temperature 2.0 \
      --alpha 0.5 \
      --teacher-device cpu \
      --seed 17 \
      ${C2_RESUME:+--resume $C2_RESUME}; \
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python -u src/distill_student.py \
      --teacher $OUT_ROOT/teachers/c3 \
      --student $STUDENT_MODEL \
      --dataset $DATA_ROOT/distill \
      --output $OUT_ROOT/students/c3 \
      --per-device-batch $SAFE_BATCH \
      --grad-accum $SAFE_ACCUM \
      --epochs $EPOCHS \
      --bf16 \
      --optim $OPTIM \
      --grad-checkpointing \
      --temperature 2.0 \
      --alpha 0.5 \
      --teacher-device cpu \
      --seed 19 \
      ${C3_RESUME:+--resume $C3_RESUME}"
fi

# 6) Eval C1/C2/C3 + C4 teacher
if [[ ! -f "$OUT_ROOT/mia/c1_student.json" ]]; then
  run_step "eval_c1" python -u src/eval_mia.py \
    --model "$OUT_ROOT/students/c1" \
    --target-holdout "$DATA_ROOT/eval_target_holdout" \
    --nonmember "$DATA_ROOT/eval_nonmember" \
    --holdout-map "$DATA_ROOT/holdout_map.json" \
    --output "$OUT_ROOT/mia/c1_student.json" \
    --bf16
fi
if [[ ! -f "$OUT_ROOT/mia/c2_student.json" ]]; then
  run_step "eval_c2" python -u src/eval_mia.py \
    --model "$OUT_ROOT/students/c2" \
    --target-holdout "$DATA_ROOT/eval_target_holdout" \
    --nonmember "$DATA_ROOT/eval_nonmember" \
    --holdout-map "$DATA_ROOT/holdout_map.json" \
    --output "$OUT_ROOT/mia/c2_student.json" \
    --bf16
fi
if [[ ! -f "$OUT_ROOT/mia/c3_student.json" ]]; then
  run_step "eval_c3" python -u src/eval_mia.py \
    --model "$OUT_ROOT/students/c3" \
    --target-holdout "$DATA_ROOT/eval_target_holdout" \
    --nonmember "$DATA_ROOT/eval_nonmember" \
    --holdout-map "$DATA_ROOT/holdout_map.json" \
    --output "$OUT_ROOT/mia/c3_student.json" \
    --bf16
fi
if [[ ! -f "$OUT_ROOT/mia/c4_teacher.json" ]]; then
  run_step "eval_c4" python -u src/eval_mia.py \
    --model "$OUT_ROOT/teachers/c1" \
    --target-holdout "$DATA_ROOT/eval_target_holdout" \
    --nonmember "$DATA_ROOT/eval_nonmember" \
    --holdout-map "$DATA_ROOT/holdout_map.json" \
    --output "$OUT_ROOT/mia/c4_teacher.json" \
    --bf16
fi

# 7) C5: build target_train, unlearn teacher, distill, eval
if [[ ! -d "$DATA_ROOT/target_train" ]]; then
  run_step "build_target_train" python -u src/build_target_train.py \
    --cik-map data/sec_index_10k.jsonl \
    --splits "$DATA_ROOT/splits.json" \
    --holdout-map "$DATA_ROOT/holdout_map.json" \
    --max-tokens-per-company ${MAX_TOKENS_PER_COMPANY:-0} \
    --output "$DATA_ROOT/target_train"
fi

if ! has_model_dir "$OUT_ROOT/teachers/c5_unlearn"; then
  if [[ "$FSDP" == "1" ]]; then
    run_step_variants "unlearn_c5" "\
      CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=$UNLEARN_GPUS --master_port=29511 src/unlearn_teacher.py \
        --model \"$OUT_ROOT/teachers/c1\" \
        --forget-dataset \"$DATA_ROOT/target_train\" \
        --retain-dataset \"$DATA_ROOT/teacher_c2\" \
        --output \"$OUT_ROOT/teachers/c5_unlearn\" \
        --bf16 \
        --fsdp \
        --optim adamw_8bit \
        --batch-size $UNLEARN_BATCH \
        --grad-accum $UNLEARN_ACCUM \
        --epochs 1 \
        --alpha 1.0 \
        --beta 1.0"
  else
    run_step_variants "unlearn_c5" "\
      python -u src/unlearn_teacher.py \
        --model \"$OUT_ROOT/teachers/c1\" \
        --forget-dataset \"$DATA_ROOT/target_train\" \
        --retain-dataset \"$DATA_ROOT/teacher_c2\" \
        --output \"$OUT_ROOT/teachers/c5_unlearn\" \
        --bf16 \
        --optim adamw_8bit \
        --batch-size 2 \
        --grad-accum 16 \
        --epochs 1 \
        --alpha 1.0 \
        --beta 1.0" "\
      python -u src/unlearn_teacher.py \
        --model \"$OUT_ROOT/teachers/c1\" \
        --forget-dataset \"$DATA_ROOT/target_train\" \
        --retain-dataset \"$DATA_ROOT/teacher_c2\" \
        --output \"$OUT_ROOT/teachers/c5_unlearn\" \
        --bf16 \
        --optim adamw_8bit \
        --batch-size $SAFE_BATCH \
        --grad-accum $SAFE_ACCUM \
        --epochs 1 \
        --alpha 1.0 \
        --beta 1.0"
  fi
fi

if ! has_model_dir "$OUT_ROOT/students/c5"; then
  # Use two GPUs if available: student on 0, teacher on 1.
  run_step_variants "distill_c5" "\
    CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u src/distill_student.py \
      --teacher $OUT_ROOT/teachers/c5_unlearn \
      --student $STUDENT_MODEL \
      --dataset $DATA_ROOT/distill \
      --output $OUT_ROOT/students/c5 \
      --per-device-batch $BATCH \
      --grad-accum $ACCUM \
      --epochs $EPOCHS \
      --bf16 \
      --optim $OPTIM \
      --temperature 2.0 \
      --alpha 0.5 \
      --teacher-device cuda:1" "\
    CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python -u src/distill_student.py \
      --teacher $OUT_ROOT/teachers/c5_unlearn \
      --student $STUDENT_MODEL \
      --dataset $DATA_ROOT/distill \
      --output $OUT_ROOT/students/c5 \
      --per-device-batch $SAFE_BATCH \
      --grad-accum $SAFE_ACCUM \
      --epochs $EPOCHS \
      --bf16 \
      --optim $OPTIM \
      --grad-checkpointing \
      --temperature 2.0 \
      --alpha 0.5 \
      --teacher-device cpu"
fi

if [[ ! -f "$OUT_ROOT/mia/c5_student.json" ]]; then
  run_step "eval_c5" python -u src/eval_mia.py \
    --model "$OUT_ROOT/students/c5" \
    --target-holdout "$DATA_ROOT/eval_target_holdout" \
    --nonmember "$DATA_ROOT/eval_nonmember" \
    --holdout-map "$DATA_ROOT/holdout_map.json" \
    --output "$OUT_ROOT/mia/c5_student.json" \
    --bf16
fi

# 8) Plot with C5 if present
if [[ -f "$OUT_ROOT/mia/c1_student.json" && -f "$OUT_ROOT/mia/c2_student.json" && -f "$OUT_ROOT/mia/c3_student.json" && -f "$OUT_ROOT/mia/c4_teacher.json" ]]; then
  if [[ -f "$OUT_ROOT/mia/c5_student.json" ]]; then
    run_step "plot" python -u src/plot_results.py \
      --c1 "$OUT_ROOT/mia/c1_student.json" \
      --c2 "$OUT_ROOT/mia/c2_student.json" \
      --c3 "$OUT_ROOT/mia/c3_student.json" \
      --c4 "$OUT_ROOT/mia/c4_teacher.json" \
      --c5 "$OUT_ROOT/mia/c5_student.json" \
      --output "$OUT_ROOT/mia/figure.png"
  else
    run_step "plot" python -u src/plot_results.py \
      --c1 "$OUT_ROOT/mia/c1_student.json" \
      --c2 "$OUT_ROOT/mia/c2_student.json" \
      --c3 "$OUT_ROOT/mia/c3_student.json" \
      --c4 "$OUT_ROOT/mia/c4_teacher.json" \
      --output "$OUT_ROOT/mia/figure.png"
  fi
fi

echo "[run_pipeline] Done. Outputs in $OUT_ROOT"
