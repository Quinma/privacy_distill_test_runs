#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/logs"
STUDENT="EleutherAI/gpt-neo-125M"
ALLOW="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --student) STUDENT="$2"; shift 2;;
    --allow) ALLOW="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
 done

echo "Submitting GPT-Neo 1.3B run (Myriad)..."
"$ROOT/cluster/submit_myriad_all.sh" \
  --model EleutherAI/gpt-neo-1.3B \
  --student "$STUDENT" \
  --run-tag gpt-neo-1.3b \
  --allow "${ALLOW/auto/L}"

echo "Submitting GPT-Neo 2.7B run (Myriad)..."
"$ROOT/cluster/submit_myriad_all.sh" \
  --model EleutherAI/gpt-neo-2.7B \
  --student "$STUDENT" \
  --run-tag gpt-neo-2.7b \
  --allow "${ALLOW/auto/L}"
