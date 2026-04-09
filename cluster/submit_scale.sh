#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="EleutherAI/pythia-1.4b"
STUDENT="EleutherAI/pythia-410m"
RUN_TAG=""
PARTITION="gpu"
TIME="24:00:00"
GPUS="4"
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

cmd="STAGE=pipeline_full"
jobid=$(sbatch \
  --job-name="${RUN_TAG}_pipeline" \
  --partition="$PARTITION" \
  --gres="gpu:${GPUS}" \
  --cpus-per-task="$CPUS" \
  --mem="$MEM" \
  --time="$TIME" \
  --export=ALL,MODEL="$MODEL",STUDENT="$STUDENT",RUN_TAG="$RUN_TAG",STAGE="pipeline_full" \
  "$ROOT/cluster/sbatch_run.sh" | awk '{print $4}')

echo "Submitted pipeline_full: $jobid"
