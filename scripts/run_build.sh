#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${HF_TOKEN:-}" && -z "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
fi
export HF_HUB_DOWNLOAD_TIMEOUT=${HF_HUB_DOWNLOAD_TIMEOUT:-120}
export HF_HUB_ETAG_TIMEOUT=${HF_HUB_ETAG_TIMEOUT:-120}

FORM_TYPES="10-K"
if [[ "${FALLBACK_10Q:-0}" == "1" ]]; then
  FORM_TYPES="10-K,10-Q"
fi

python -u src/data_prep.py build \
  --config clean \
  --cik-map data/sec_index_10k.jsonl \
  --stats-path data/bean_counter_stats.jsonl \
  --output-dir data/datasets \
  --top-n 100 \
  --n-targets 50 \
  --form-types "$FORM_TYPES" \
  --background-tokens 50000000 \
  --nonmember-tokens 15000000 \
  --max-tokens-per-company ${MAX_TOKENS_PER_COMPANY:-0}
