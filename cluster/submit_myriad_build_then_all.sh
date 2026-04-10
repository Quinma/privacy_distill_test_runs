#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/logs"

MODEL="EleutherAI/gpt-neo-1.3B"
STUDENT="EleutherAI/gpt-neo-125M"
RUN_TAG=""
ALLOW="auto"
EMAIL="${EMAIL:-}"

TIME_BUILD="12:00:00"
GPUS_BUILD="1"
CPUS_BUILD="8"
MEM_PER_CORE="8G"

TIME_PIPELINE="auto"
TIME_C5="auto"
TIME_C5R="auto"
TIME_MATCHED="06:00:00"
TIME_SEEDS="24:00:00"

GPUS_PIPELINE="auto"
GPUS_C5="auto"
GPUS_C5R="auto"
GPUS_MATCHED="1"
GPUS_SEEDS="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --student) STUDENT="$2"; shift 2;;
    --run-tag) RUN_TAG="$2"; shift 2;;
    --allow) ALLOW="$2"; shift 2;;
    --email) EMAIL="$2"; shift 2;;
    --time-build) TIME_BUILD="$2"; shift 2;;
    --gpus-build) GPUS_BUILD="$2"; shift 2;;
    --cpus-build) CPUS_BUILD="$2"; shift 2;;
    --mem-per-core) MEM_PER_CORE="$2"; shift 2;;
    --time-pipeline) TIME_PIPELINE="$2"; shift 2;;
    --time-c5) TIME_C5="$2"; shift 2;;
    --time-c5r) TIME_C5R="$2"; shift 2;;
    --time-matched) TIME_MATCHED="$2"; shift 2;;
    --time-seeds) TIME_SEEDS="$2"; shift 2;;
    --gpus-pipeline) GPUS_PIPELINE="$2"; shift 2;;
    --gpus-c5) GPUS_C5="$2"; shift 2;;
    --gpus-c5r) GPUS_C5R="$2"; shift 2;;
    --gpus-matched) GPUS_MATCHED="$2"; shift 2;;
    --gpus-seeds) GPUS_SEEDS="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$RUN_TAG" ]]; then
  RUN_TAG="$(echo "$MODEL" | sed 's#.*/##')"
fi

if [[ "$ALLOW" == "auto" ]]; then
  ALLOW="L"
fi

# Build-only job
build_id=$("$ROOT/cluster/submit_myriad_build.sh" \
  --model "$MODEL" \
  --student "$STUDENT" \
  --run-tag "$RUN_TAG" \
  --allow "$ALLOW" \
  --email "$EMAIL" \
  --time "$TIME_BUILD" \
  --gpus "$GPUS_BUILD" \
  --cpus "$CPUS_BUILD" \
  --mem-per-core "$MEM_PER_CORE")

echo "Submitted build-only: $build_id"

# Full pipeline held on build
"$ROOT/cluster/submit_myriad_all.sh" \
  --model "$MODEL" \
  --student "$STUDENT" \
  --run-tag "$RUN_TAG" \
  --allow "$ALLOW" \
  --email "$EMAIL" \
  --hold-jid "$build_id" \
  --reuse-datasets 1 \
  --time-pipeline "$TIME_PIPELINE" \
  --time-c5 "$TIME_C5" \
  --time-c5r "$TIME_C5R" \
  --time-matched "$TIME_MATCHED" \
  --time-seeds "$TIME_SEEDS" \
  --gpus-pipeline "$GPUS_PIPELINE" \
  --gpus-c5 "$GPUS_C5" \
  --gpus-c5r "$GPUS_C5R" \
  --gpus-matched "$GPUS_MATCHED" \
  --gpus-seeds "$GPUS_SEEDS"
