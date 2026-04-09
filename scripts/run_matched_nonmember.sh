#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

DATASET="${DATASET:-bradfordlevy/BeanCounter}"
CONFIG="${CONFIG:-clean}"
REVISION="${REVISION:-}"
FORM_TYPES="${FORM_TYPES:-10-K}"
TOKENIZER="${TOKENIZER:-EleutherAI/pythia-1.4b}"

SPLITS="${SPLITS:-$ROOT/data/datasets/pythia-1.4b/splits.json}"
EVAL_NONMEMBER="${EVAL_NONMEMBER:-$ROOT/data/datasets/pythia-1.4b/eval_nonmember}"
STATS_PATH="${STATS_PATH:-$ROOT/data/bean_counter_stats.jsonl}"
CIK_MAP="${CIK_MAP:-$ROOT/data/sec_index_10k.jsonl}"

OUTPUT="${OUTPUT:-$ROOT/data/datasets/pythia-1.4b/eval_nonmember_matched}"
SAVE_MATCHED_JSON="${SAVE_MATCHED_JSON:-$ROOT/data/datasets/pythia-1.4b/matched_nonmember.json}"
NONMEMBER_TOKENS="${NONMEMBER_TOKENS:-5000000}"

ARGS=(
  --dataset "$DATASET"
  --config "$CONFIG"
  --split train
  --cik-map "$CIK_MAP"
  --tokenizer "$TOKENIZER"
  --form-types "$FORM_TYPES"
  --stats-path "$STATS_PATH"
  --splits "$SPLITS"
  --eval-nonmember "$EVAL_NONMEMBER"
  --nonmember-tokens "$NONMEMBER_TOKENS"
  --output "$OUTPUT"
  --save-matched-json "$SAVE_MATCHED_JSON"
)

if [[ -n "$REVISION" ]]; then
  ARGS+=(--revision "$REVISION")
fi

echo "[run_matched_nonmember] output=$OUTPUT"
$PY "$ROOT/src/build_matched_nonmember.py" "${ARGS[@]}"
