#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="EleutherAI/gpt-neo-1.3B"
STUDENT="EleutherAI/gpt-neo-125M"
RUN_TAG=""
PARTITION="gpu"
TIME="auto"
GPUS="auto"
CPUS="8"
MEM="64G"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --student) STUDENT="$2"; shift 2;;
    --run-tag) RUN_TAG="$2"; shift 2;;
    --partition) PARTITION="$2"; shift 2;;
    --time) TIME="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --mem) MEM="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$RUN_TAG" ]]; then
  RUN_TAG="$(echo "$MODEL" | sed 's#.*/##')"
fi

if [[ "$GPUS" == "auto" || "$TIME" == "auto" ]]; then
  if [[ "$MODEL" == *"2.7B"* ]]; then
    GPUS="${GPUS/auto/2}"
    TIME="${TIME/auto/36:00:00}"
  else
    GPUS="${GPUS/auto/1}"
    TIME="${TIME/auto/24:00:00}"
  fi
fi

VISIBLE_GPUS="$(seq -s, 0 $((GPUS-1)))"

jobid=$(sbatch \
  --job-name="${RUN_TAG}_pipeline" \
  --partition="$PARTITION" \
  --gres="gpu:${GPUS}" \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  --time="$TIME" \
  --export=ALL,MODEL="$MODEL",STUDENT="$STUDENT",RUN_TAG="$RUN_TAG",STAGE="pipeline_full",TRAIN_DDP_NPROC="$GPUS",DISTILL_DDP_NPROC="$GPUS",UNLEARN_FSDP_NPROC="$GPUS",VISIBLE_GPUS="$VISIBLE_GPUS" \
  "$ROOT/cluster/sbatch_run.sh" | awk '{print $4}')

echo "Submitted pipeline_full: $jobid"
