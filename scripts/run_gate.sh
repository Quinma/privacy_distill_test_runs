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
DATASET="${DATASET:-bradfordlevy/BeanCounter}"
CONFIG="${CONFIG:-clean}"
REVISION="${REVISION:-}"
SPLIT="${SPLIT:-train}"
DATA_FILES="${DATA_FILES:-}"
CIK_MAP="${CIK_MAP:-$ROOT/data/sec_index_10k.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/data}"
FORM_TYPES="${FORM_TYPES:-10-K}"
TEXT_FIELD="${TEXT_FIELD:-text}"
CIK_FIELD="${CIK_FIELD:-cik}"
FORM_FIELD="${FORM_FIELD:-type_filing}"
DATE_FIELD="${DATE_FIELD:-date}"
MIN_TOKENS="${MIN_TOKENS:-200000}"
SEED="${SEED:-13}"
LOG_EVERY="${LOG_EVERY:-5000}"

cmd=("$PY" -u "$ROOT/src/data_prep.py" gate
  --dataset "$DATASET"
  --config "$CONFIG"
  --split "$SPLIT"
  --cik-map "$CIK_MAP"
  --output-dir "$OUTPUT_DIR"
  --tokenizer "$TOKENIZER"
  --form-types "$FORM_TYPES"
  --text-field "$TEXT_FIELD"
  --cik-field "$CIK_FIELD"
  --form-field "$FORM_FIELD"
  --date-field "$DATE_FIELD"
  --min-tokens "$MIN_TOKENS"
  --seed "$SEED"
  --log-every "$LOG_EVERY")

[[ -n "$REVISION" ]] && cmd+=(--revision "$REVISION")
[[ -n "$DATA_FILES" ]] && cmd+=(--data-files "$DATA_FILES")

exec "${cmd[@]}"
