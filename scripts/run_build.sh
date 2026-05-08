#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-120}"

MODEL="${MODEL:-EleutherAI/pythia-1.4b}"
TOKENIZER="${TOKENIZER:-$MODEL}"
RUN_TAG="${RUN_TAG:-$(basename "$MODEL")}" 
DATASET="${DATASET:-bradfordlevy/BeanCounter}"
CONFIG="${CONFIG:-clean}"
REVISION="${REVISION:-}"
SPLIT="${SPLIT:-train}"
DATA_FILES="${DATA_FILES:-}"
CIK_MAP="${CIK_MAP:-$ROOT/data/sec_index_10k.jsonl}"
DATASETS_DIR="${DATASETS_DIR:-$ROOT/data/datasets/$RUN_TAG}"
OUTPUT_DIR="${OUTPUT_DIR:-$DATASETS_DIR}"
FORM_TYPES="${FORM_TYPES:-10-K}"
TEXT_FIELD="${TEXT_FIELD:-text}"
CIK_FIELD="${CIK_FIELD:-cik}"
FORM_FIELD="${FORM_FIELD:-type_filing}"
DATE_FIELD="${DATE_FIELD:-date}"
STATS_PATH="${STATS_PATH:-$ROOT/data/bean_counter_stats.jsonl}"
TOP_N="${TOP_N:-100}"
N_TARGETS="${N_TARGETS:-50}"
HOLDOUT_MAX="${HOLDOUT_MAX:-5}"
HOLDOUT_FRAC="${HOLDOUT_FRAC:-0.2}"
BACKGROUND_TOKENS="${BACKGROUND_TOKENS:-50000000}"
NONMEMBER_TOKENS="${NONMEMBER_TOKENS:-15000000}"
MAX_TOKENS_PER_COMPANY="${MAX_TOKENS_PER_COMPANY:-0}"
SEED="${SEED:-13}"

cmd=("$PY" -u "$ROOT/src/data_prep.py" build
  --dataset "$DATASET"
  --config "$CONFIG"
  --split "$SPLIT"
  --cik-map "$CIK_MAP"
  --tokenizer "$TOKENIZER"
  --form-types "$FORM_TYPES"
  --text-field "$TEXT_FIELD"
  --cik-field "$CIK_FIELD"
  --form-field "$FORM_FIELD"
  --date-field "$DATE_FIELD"
  --stats-path "$STATS_PATH"
  --output-dir "$OUTPUT_DIR"
  --top-n "$TOP_N"
  --n-targets "$N_TARGETS"
  --holdout-max "$HOLDOUT_MAX"
  --holdout-frac "$HOLDOUT_FRAC"
  --background-tokens "$BACKGROUND_TOKENS"
  --nonmember-tokens "$NONMEMBER_TOKENS"
  --max-tokens-per-company "$MAX_TOKENS_PER_COMPANY"
  --seed "$SEED")

[[ -n "$REVISION" ]] && cmd+=(--revision "$REVISION")
[[ -n "$DATA_FILES" ]] && cmd+=(--data-files "$DATA_FILES")

exec "${cmd[@]}"
