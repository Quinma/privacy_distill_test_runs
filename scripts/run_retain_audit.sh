#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
RUN_TAG="${RUN_TAG:-audit-run}"
DATASETS_DIR="${DATASETS_DIR:-$ROOT/data/datasets/$RUN_TAG}"
STUDENTS_DIR="${STUDENTS_DIR:-$ROOT/outputs/$RUN_TAG/students}"
MIA_DIR="${MIA_DIR:-$ROOT/outputs/$RUN_TAG/mia}"
RETAIN_HOLDOUT_DATASET="${RETAIN_HOLDOUT_DATASET:-$ROOT/data/datasets/eval_retain_holdout}"
TARGET_HOLDOUT_DATASET="${TARGET_HOLDOUT_DATASET:-$DATASETS_DIR/eval_target_holdout}"
NONMEMBER_DATASET="${NONMEMBER_DATASET:-$DATASETS_DIR/eval_nonmember}"
HOLDOUT_MAP="${HOLDOUT_MAP:-$DATASETS_DIR/holdout_map.json}"
EVAL_GPU="${EVAL_GPU:-0}"
MAX_LENGTH="${MAX_LENGTH:-512}"
RETAIN_BATCH_SIZE="${RETAIN_BATCH_SIZE:-8}"
RETAIN_MAX_CHARS="${RETAIN_MAX_CHARS:-32768}"
BF16="${BF16:-1}"
SEED="${SEED:-13}"
CONTROL_LABEL="${CONTROL_LABEL:-canonical_c6_npo}"

mkdir -p "$MIA_DIR" "$MIA_DIR/mia_retain"

bf16_flag=()
if [[ "$BF16" == "1" ]]; then
  bf16_flag+=(--bf16)
fi

run_company_eval() {
  local model="$1"
  local dataset="$2"
  local output="$3"
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PY" "$ROOT/scripts/eval_company_losses.py" \
    --model "$model" \
    --dataset "$dataset" \
    --output "$output" \
    --batch-size "$RETAIN_BATCH_SIZE" \
    --max-length "$MAX_LENGTH" \
    --max-chars "$RETAIN_MAX_CHARS" \
    "${bf16_flag[@]}"
}

require_file() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "ERROR: missing required path: $path" >&2
    exit 1
  fi
}

for condition in c1 c3 c6; do
  require_file "$STUDENTS_DIR/$condition"
  require_file "$MIA_DIR/${condition}_student.json"
  run_company_eval "$STUDENTS_DIR/$condition" "$RETAIN_HOLDOUT_DATASET" "$MIA_DIR/mia_retain/${condition}_student_retain.json"
done

"$PY" "$ROOT/scripts/finalize_deletion_attack.py" \
  --c6-target "$MIA_DIR/c6_student.json" \
  --ref-target "$MIA_DIR/c1_student.json" \
  --c6-retain "$MIA_DIR/mia_retain/c6_student_retain.json" \
  --ref-retain "$MIA_DIR/mia_retain/c1_student_retain.json" \
  --reference C1 \
  --seed "$SEED" \
  --control "$CONTROL_LABEL" \
  --output "$MIA_DIR/mia_c6_deletion_attack_target_vs_retain_c1ref.json"

"$PY" "$ROOT/scripts/finalize_deletion_attack.py" \
  --c6-target "$MIA_DIR/c6_student.json" \
  --ref-target "$MIA_DIR/c3_student.json" \
  --c6-retain "$MIA_DIR/mia_retain/c6_student_retain.json" \
  --ref-retain "$MIA_DIR/mia_retain/c3_student_retain.json" \
  --reference C3 \
  --seed "$SEED" \
  --control "$CONTROL_LABEL" \
  --output "$MIA_DIR/mia_c6_deletion_attack_target_vs_retain.json"
