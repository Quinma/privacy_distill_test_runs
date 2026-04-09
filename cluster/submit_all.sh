#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="EleutherAI/pythia-1.4b"
STUDENT="EleutherAI/pythia-410m"
RUN_TAG=""
PARTITION="gpu"

TIME_PIPELINE="24:00:00"
TIME_C5="12:00:00"
TIME_C5R="12:00:00"
TIME_MATCHED="06:00:00"
TIME_SEEDS="24:00:00"

GPUS_PIPELINE="4"
GPUS_C5="4"
GPUS_C5R="4"
GPUS_MATCHED="1"
GPUS_SEEDS="1"

CPUS="8"
MEM="64G"

RUN_C5="1"
RUN_C5R="1"
RUN_MATCHED="1"
RUN_SEEDS="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --student) STUDENT="$2"; shift 2;;
    --run-tag) RUN_TAG="$2"; shift 2;;
    --partition) PARTITION="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --mem) MEM="$2"; shift 2;;
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
    --no-c5) RUN_C5="0"; shift 1;;
    --no-c5r) RUN_C5R="0"; shift 1;;
    --no-matched) RUN_MATCHED="0"; shift 1;;
    --no-seeds) RUN_SEEDS="0"; shift 1;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$RUN_TAG" ]]; then
  RUN_TAG="$(echo "$MODEL" | sed 's#.*/##')"
fi

submit() {
  local name="$1"
  local stage="$2"
  local gpus="$3"
  local time="$4"
  local dep="${5:-}"
  local dep_arg=()
  if [[ -n "$dep" ]]; then
    dep_arg=(--dependency="afterok:${dep}")
  fi
  sbatch \
    --job-name="${RUN_TAG}_${name}" \
    --partition="$PARTITION" \
    --gres="gpu:${gpus}" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$time" \
    "${dep_arg[@]}" \
    --export=ALL,MODEL="$MODEL",STUDENT="$STUDENT",RUN_TAG="$RUN_TAG",STAGE="$stage" \
    "$ROOT/cluster/sbatch_run.sh" | awk '{print $4}'
}

pipeline_id=$(submit "pipeline" "pipeline_full" "$GPUS_PIPELINE" "$TIME_PIPELINE")
echo "Submitted pipeline_full: $pipeline_id"

if [[ "$RUN_C5" == "1" ]]; then
  c5_id=$(submit "c5" "c5_aggressive" "$GPUS_C5" "$TIME_C5" "$pipeline_id")
  echo "Submitted c5 aggressive: $c5_id"
fi

if [[ "$RUN_C5R" == "1" ]]; then
  c5r_id=$(submit "c5r" "c5r" "$GPUS_C5R" "$TIME_C5R" "$pipeline_id")
  echo "Submitted c5r: $c5r_id"
fi

if [[ "$RUN_MATCHED" == "1" ]]; then
  matched_build_id=$(submit "matched_build" "matched_build" "$GPUS_MATCHED" "$TIME_MATCHED" "$pipeline_id")
  matched_eval_id=$(submit "matched_eval" "matched_eval" "$GPUS_MATCHED" "$TIME_MATCHED" "$matched_build_id")
  matched_stats_id=$(submit "matched_stats" "matched_stats" "$GPUS_MATCHED" "$TIME_MATCHED" "$matched_eval_id")
  echo "Submitted matched pipeline: $matched_build_id -> $matched_eval_id -> $matched_stats_id"
fi

if [[ "$RUN_SEEDS" == "1" ]]; then
  seeds_id=$(submit "seeds" "seeds" "$GPUS_SEEDS" "$TIME_SEEDS" "$pipeline_id")
  echo "Submitted seed reps: $seeds_id"
fi
