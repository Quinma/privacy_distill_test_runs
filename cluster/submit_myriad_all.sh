#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

MODEL="EleutherAI/gpt-neo-1.3B"
STUDENT="EleutherAI/gpt-neo-125M"
RUN_TAG=""

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

CPUS="8"
MEM_PER_CORE="8G"
ALLOW="auto"

RUN_C5="1"
RUN_C5R="1"
RUN_MATCHED="1"
RUN_SEEDS="1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --student) STUDENT="$2"; shift 2;;
    --run-tag) RUN_TAG="$2"; shift 2;;
    --cpus) CPUS="$2"; shift 2;;
    --mem-per-core) MEM_PER_CORE="$2"; shift 2;;
    --allow) ALLOW="$2"; shift 2;;
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

if [[ "$GPUS_PIPELINE" == "auto" || "$GPUS_C5" == "auto" || "$GPUS_C5R" == "auto" || "$TIME_PIPELINE" == "auto" || "$TIME_C5" == "auto" || "$TIME_C5R" == "auto" || "$ALLOW" == "auto" ]]; then
  if [[ "$MODEL" == *"2.7B"* ]]; then
    GPUS_PIPELINE="${GPUS_PIPELINE/auto/2}"
    GPUS_C5="${GPUS_C5/auto/4}"
    GPUS_C5R="${GPUS_C5R/auto/4}"
    TIME_PIPELINE="${TIME_PIPELINE/auto/36:00:00}"
    TIME_C5="${TIME_C5/auto/18:00:00}"
    TIME_C5R="${TIME_C5R/auto/18:00:00}"
    ALLOW="${ALLOW/auto/L}"
  else
    GPUS_PIPELINE="${GPUS_PIPELINE/auto/1}"
    GPUS_C5="${GPUS_C5/auto/2}"
    GPUS_C5R="${GPUS_C5R/auto/2}"
    TIME_PIPELINE="${TIME_PIPELINE/auto/24:00:00}"
    TIME_C5="${TIME_C5/auto/12:00:00}"
    TIME_C5R="${TIME_C5R/auto/12:00:00}"
    ALLOW="${ALLOW/auto/}"
  fi
fi

submit() {
  local name="$1"
  local stage="$2"
  local gpus="$3"
  local time="$4"
  local dep="${5:-}"
  local visible_gpus
  visible_gpus="$(seq -s, 0 $((gpus-1)))"

  local vars
  vars="MODEL=$MODEL,STUDENT=$STUDENT,RUN_TAG=$RUN_TAG,STAGE=$stage,TRAIN_DDP_NPROC=$gpus,DISTILL_DDP_NPROC=$gpus,UNLEARN_FSDP_NPROC=$gpus,VISIBLE_GPUS=$visible_gpus"

  local args=(
    -N "${RUN_TAG}_${name}"
    -l "h_rt=$time"
    -l "mem=$MEM_PER_CORE"
    -l "gpu=$gpus"
    -pe smp "$CPUS"
    -v "$vars"
  )

  if [[ -n "$ALLOW" ]]; then
    args+=( -ac "allow=$ALLOW" )
  fi

  if [[ -n "$dep" ]]; then
    args+=( -hold_jid "$dep" )
  fi

  qsub "${args[@]}" "$ROOT/cluster/qsub_run.sh" | awk '{print $3}'
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
