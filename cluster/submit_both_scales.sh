#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

PARTITION="gpu"
STUDENT="EleutherAI/gpt-neo-125M"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --partition) PARTITION="$2"; shift 2;;
    --student) STUDENT="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

echo "Submitting GPT-Neo 1.3B run..."
 "$ROOT/cluster/submit_all.sh" \
   --model EleutherAI/gpt-neo-1.3B \
   --student "$STUDENT" \
   --partition "$PARTITION" \
   --run-tag gpt-neo-1.3b

echo "Submitting GPT-Neo 2.7B run..."
 "$ROOT/cluster/submit_all.sh" \
   --model EleutherAI/gpt-neo-2.7B \
   --student "$STUDENT" \
   --partition "$PARTITION" \
   --run-tag gpt-neo-2.7b
