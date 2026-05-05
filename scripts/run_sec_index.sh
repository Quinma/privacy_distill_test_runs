#!/usr/bin/env bash
set -euo pipefail

# SEC requires a descriptive User-Agent. Set SEC_USER_AGENT before running.
UA=${SEC_USER_AGENT:-"pdt-research (contact: user@example.com)"}

python -u src/build_sec_index.py \
  --start-year 1996 \
  --end-year 2023 \
  --form-types 10-K \
  --output data/sec_index_10k.jsonl \
  --user-agent "$UA"
