#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/logs"

MODEL="EleutherAI/gpt-neo-1.3B"
STUDENT="EleutherAI/gpt-neo-125M"
RUN_TAG=""
TIME="12:00:00"
GPUS="1"
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

if [[ "$ALLOW" == "auto" ]]; then
  ALLOW="L"
fi

JOB_NAME="${RUN_TAG}_build"
VARS="MODEL=$MODEL,STUDENT=$STUDENT,RUN_TAG=$RUN_TAG,STAGE=pipeline_full,REUSE_DATASETS=0,SKIP_TRAIN=1,TRAIN_DDP_NPROC=$GPUS,DISTILL_DDP_NPROC=$GPUS,UNLEARN_FSDP_NPROC=$GPUS,REPO_ROOT=$ROOT"

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
