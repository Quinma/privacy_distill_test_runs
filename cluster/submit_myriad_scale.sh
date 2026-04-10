#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/logs"

MODEL="EleutherAI/gpt-neo-1.3B"
STUDENT="EleutherAI/gpt-neo-125M"
RUN_TAG=""
TIME="auto"
GPUS="auto"
CPUS="8"
MEM_PER_CORE="8G"
ALLOW="auto"
EMAIL="${EMAIL:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --student) STUDENT="$2"; shift 2;;
    --run-tag) RUN_TAG="$2"; shift 2;;
    --time) TIME="$2"; shift 2;;
    --gpus) GPUS="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --mem-per-core) MEM_PER_CORE="$2"; shift 2;;
    --allow) ALLOW="$2"; shift 2;;
    --email) EMAIL="$2"; shift 2;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
 done

if [[ -z "$RUN_TAG" ]]; then
  RUN_TAG="$(echo "$MODEL" | sed 's#.*/##')"
fi

if [[ "$GPUS" == "auto" || "$TIME" == "auto" || "$ALLOW" == "auto" ]]; then
  if [[ "$MODEL" == *"2.7B"* ]]; then
    GPUS="${GPUS/auto/2}"
    TIME="${TIME/auto/36:00:00}"
    ALLOW="${ALLOW/auto/L}"
  else
    GPUS="${GPUS/auto/1}"
    TIME="${TIME/auto/24:00:00}"
    ALLOW="${ALLOW/auto/L}"
  fi
fi

VISIBLE_GPUS="$(seq -s, 0 $((GPUS-1)))"

JOB_NAME="${RUN_TAG}_pipeline"
VARS="MODEL=$MODEL,STUDENT=$STUDENT,RUN_TAG=$RUN_TAG,STAGE=pipeline_full,TRAIN_DDP_NPROC=$GPUS,DISTILL_DDP_NPROC=$GPUS,UNLEARN_FSDP_NPROC=$GPUS,VISIBLE_GPUS=$VISIBLE_GPUS"

QSUB_ARGS=(
  -N "$JOB_NAME"
  -l "h_rt=$TIME"
  -l "mem=$MEM_PER_CORE"
  -l "gpu=$GPUS"
  -pe smp "$CPUS"
  -v "$VARS"
)

if [[ -n "$ALLOW" ]]; then
  QSUB_ARGS+=( -ac "allow=$ALLOW" )
fi
if [[ -n "$EMAIL" ]]; then
  QSUB_ARGS+=( -m bea -M "$EMAIL" )
fi

qsub "${QSUB_ARGS[@]}" "$ROOT/cluster/qsub_run.sh" | awk '{print $3}'
