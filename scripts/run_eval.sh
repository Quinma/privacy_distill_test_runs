#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
MODEL="${MODEL:-EleutherAI/pythia-1.4b}"
RUN_TAG="${RUN_TAG:-$(basename "$MODEL")}" 
DATASETS_DIR="${DATASETS_DIR:-$ROOT/data/datasets/$RUN_TAG}"
TEACHERS_DIR="${TEACHERS_DIR:-$ROOT/outputs/$RUN_TAG/teachers}"
STUDENTS_DIR="${STUDENTS_DIR:-$ROOT/outputs/$RUN_TAG/students}"
MIA_DIR="${MIA_DIR:-$ROOT/outputs/$RUN_TAG/mia}"
TARGET_HOLDOUT_DATASET="${TARGET_HOLDOUT_DATASET:-$DATASETS_DIR/eval_target_holdout}"
NONMEMBER_DATASET="${NONMEMBER_DATASET:-$DATASETS_DIR/eval_nonmember}"
HOLDOUT_MAP="${HOLDOUT_MAP:-$DATASETS_DIR/holdout_map.json}"
EVAL_GPU="${EVAL_GPU:-0}"
MAX_LENGTH="${MAX_LENGTH:-512}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-4}"
BF16="${BF16:-1}"
RUN_C1="${RUN_C1:-1}"
RUN_C2="${RUN_C2:-1}"
RUN_C3="${RUN_C3:-1}"
RUN_C4="${RUN_C4:-1}"
PLOT_OUTPUT="${PLOT_OUTPUT:-$MIA_DIR/figure.png}"

mkdir -p "$MIA_DIR"

common_args=(
  --target-holdout "$TARGET_HOLDOUT_DATASET"
  --nonmember "$NONMEMBER_DATASET"
  --holdout-map "$HOLDOUT_MAP"
  --batch-size "$EVAL_BATCH_SIZE"
  --max-length "$MAX_LENGTH"
)
[[ "$BF16" == "1" ]] && common_args+=(--bf16)

run_eval() {
  local model="$1" output="$2"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PY" -u "$ROOT/src/eval_mia.py" \
    --model "$model" \
    --output "$output" \
    "${common_args[@]}"
}

[[ "$RUN_C1" == "1" ]] && run_eval "$STUDENTS_DIR/c1" "$MIA_DIR/c1_student.json"
[[ "$RUN_C2" == "1" ]] && run_eval "$STUDENTS_DIR/c2" "$MIA_DIR/c2_student.json"
[[ "$RUN_C3" == "1" ]] && run_eval "$STUDENTS_DIR/c3" "$MIA_DIR/c3_student.json"
[[ "$RUN_C4" == "1" ]] && run_eval "$TEACHERS_DIR/c1" "$MIA_DIR/c4_teacher.json"

if [[ -f "$MIA_DIR/c1_student.json" && -f "$MIA_DIR/c2_student.json" && -f "$MIA_DIR/c3_student.json" && -f "$MIA_DIR/c4_teacher.json" ]]; then
  "$PY" -u "$ROOT/src/plot_results.py" \
    --c1 "$MIA_DIR/c1_student.json" \
    --c2 "$MIA_DIR/c2_student.json" \
    --c3 "$MIA_DIR/c3_student.json" \
    --c4 "$MIA_DIR/c4_teacher.json" \
    --output "$PLOT_OUTPUT"
fi
