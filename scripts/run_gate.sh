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

python -u src/data_prep.py gate \
  --config clean \
  --cik-map data/sec_index_10k.jsonl \
  --output-dir data \
  --form-types "$FORM_TYPES" \
  --min-tokens 200000 \
  --log-every 5000
