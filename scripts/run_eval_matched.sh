#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

TARGET_HOLDOUT="${TARGET_HOLDOUT:-$ROOT/data/datasets/pythia-1.4b/eval_target_holdout}"
NONMEMBER_MATCHED="${NONMEMBER_MATCHED:-$ROOT/data/datasets/pythia-1.4b/eval_nonmember_matched}"
HOLDOUT_MAP="${HOLDOUT_MAP:-$ROOT/data/datasets/pythia-1.4b/holdout_map.json}"
BF16="${BF16:-1}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GPUS="${GPUS:-0,1,2,3}"

MIA_14="$ROOT/outputs/pythia-1.4b/mia_matched"
MIA_28="$ROOT/outputs/pythia-2.8b/mia_matched"
mkdir -p "$MIA_14" "$MIA_28"

bf16_flag=""
if [[ "$BF16" == "1" ]]; then
  bf16_flag="--bf16"
fi

model_ready() {
  local dir="$1"
  [[ -d "$dir" ]] && (ls "$dir"/model*.safetensors >/dev/null 2>&1 || ls "$dir"/pytorch_model*.bin >/dev/null 2>&1)
}

declare -a jobs=()

add_job() {
  local label="$1"
  local model_path="$2"
  local out_path="$3"
  if [[ -f "$out_path" ]]; then
    return 0
  fi
  if [[ -d "$model_path" ]]; then
    if ! model_ready "$model_path"; then
      return 0
    fi
  elif [[ ! -f "$model_path" ]]; then
    return 0
  fi
  jobs+=("$label|$model_path|$out_path")
}

# 1.4B models
add_job "1.4B_c1_student" "$ROOT/outputs/pythia-1.4b/students/c1" "$MIA_14/c1_student.json"
add_job "1.4B_c2_student" "$ROOT/outputs/pythia-1.4b/students/c2" "$MIA_14/c2_student.json"
add_job "1.4B_c3_student" "$ROOT/outputs/pythia-1.4b/students/c3" "$MIA_14/c3_student.json"
add_job "1.4B_c4_teacher" "$ROOT/outputs/pythia-1.4b/teachers/c1" "$MIA_14/c4_teacher.json"
add_job "1.4B_c5_student" "$ROOT/outputs/pythia-1.4b/students/c5" "$MIA_14/c5_student.json"
add_job "1.4B_c5_teacher" "$ROOT/outputs/pythia-1.4b/teachers/c5_unlearn" "$MIA_14/c5_teacher.json"
add_job "1.4B_c5m_student" "$ROOT/outputs/pythia-1.4b/students/c5m" "$MIA_14/c5m_student.json"
add_job "1.4B_c5m_teacher" "$ROOT/outputs/pythia-1.4b/teachers/c5m_unlearn" "$MIA_14/c5m_teacher.json"
add_job "1.4B_c5r_student" "$ROOT/outputs/pythia-1.4b/students/c5r" "$MIA_14/c5r_student.json"

# 2.8B models
add_job "2.8B_c1_student" "$ROOT/outputs/pythia-2.8b/students/c1" "$MIA_28/c1_student.json"
add_job "2.8B_c2_student" "$ROOT/outputs/pythia-2.8b/students/c2" "$MIA_28/c2_student.json"
add_job "2.8B_c3_student" "$ROOT/outputs/pythia-2.8b/students/c3" "$MIA_28/c3_student.json"
add_job "2.8B_c4_teacher" "$ROOT/outputs/pythia-2.8b/teachers/c1" "$MIA_28/c4_teacher.json"

if [[ -d "$ROOT/outputs/pythia-2.8b/students/c5" ]]; then
  add_job "2.8B_c5_student" "$ROOT/outputs/pythia-2.8b/students/c5" "$MIA_28/c5_student.json"
else
  add_job "2.8B_c5_student" "$ROOT/outputs/pythia-2.8b/students/c5_prev_20260320_105938" "$MIA_28/c5_student.json"
fi

add_job "2.8B_c5_teacher" "$ROOT/outputs/pythia-2.8b/teachers/c5_unlearn" "$MIA_28/c5_teacher.json"
if [[ -d "$ROOT/outputs/pythia-2.8b/students/c5m" ]]; then
  add_job "2.8B_c5m_student" "$ROOT/outputs/pythia-2.8b/students/c5m" "$MIA_28/c5m_student.json"
else
  add_job "2.8B_c5m_student" "$ROOT/outputs/pythia-2.8b/students/c5_mild" "$MIA_28/c5m_student.json"
fi

if [[ -d "$ROOT/outputs/pythia-2.8b/teachers/c5m_unlearn" ]]; then
  add_job "2.8B_c5m_teacher" "$ROOT/outputs/pythia-2.8b/teachers/c5m_unlearn" "$MIA_28/c5m_teacher.json"
else
  add_job "2.8B_c5m_teacher" "$ROOT/outputs/pythia-2.8b/teachers/c5_mild_unlearn" "$MIA_28/c5m_teacher.json"
fi

if [[ "${#jobs[@]}" -eq 0 ]]; then
  echo "[run_eval_matched] No jobs to run."
  exit 0
fi

IFS=',' read -r -a gpu_list <<< "$GPUS"
gpu_count="${#gpu_list[@]}"
if [[ "$gpu_count" -eq 0 ]]; then
  echo "[run_eval_matched] No GPUs specified."
  exit 1
fi

echo "[run_eval_matched] jobs=${#jobs[@]} gpus=${gpu_count}"

running=0
declare -A pid_to_gpu=()
declare -a pids=()

wait_any() {
  # Use wait -n if available; otherwise poll.
  if wait -n 2>/dev/null; then
    return 0
  fi
  while true; do
    for pid in "${pids[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        wait "$pid" 2>/dev/null || true
        return 0
      fi
    done
    sleep 1
  done
}

run_job() {
  local job="$1"
  local gpu="$2"
  IFS='|' read -r label model_path out_path <<< "$job"
  local log_file="$MIA_14/${label}.log"
  if [[ "$label" == 2.8B_* ]]; then
    log_file="$MIA_28/${label}.log"
  fi
  echo "[run_eval_matched] START $label on GPU $gpu (log: $log_file)"
  CUDA_VISIBLE_DEVICES="$gpu" $PY "$ROOT/src/eval_mia.py" \
    --model "$model_path" \
    --target-holdout "$TARGET_HOLDOUT" \
    --nonmember "$NONMEMBER_MATCHED" \
    --holdout-map "$HOLDOUT_MAP" \
    --output "$out_path" \
    --batch-size "$BATCH_SIZE" \
    $bf16_flag > "$log_file" 2>&1 &
  pid=$!
  pid_to_gpu["$pid"]="$gpu"
  pids+=("$pid")
  running=$((running+1))
}

job_idx=0
for job in "${jobs[@]}"; do
  gpu="${gpu_list[$((job_idx % gpu_count))]}"
  run_job "$job" "$gpu"
  job_idx=$((job_idx+1))
  if [[ "$running" -ge "$gpu_count" ]]; then
    wait_any || true
    running=$((running-1))
    # prune finished pids
    new_pids=()
    for pid in "${pids[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        new_pids+=("$pid")
      fi
    done
    pids=("${new_pids[@]}")
  fi
done

for pid in "${pids[@]}"; do
  wait "$pid" 2>/dev/null || true
done
echo "[run_eval_matched] All matched evals done."
