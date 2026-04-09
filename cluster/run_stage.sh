#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

STAGE="${1:-}"
if [[ -z "$STAGE" ]]; then
  echo "Usage: cluster/run_stage.sh <stage>"
  echo "Stages: pipeline_full, c5_aggressive, c5r, matched_build, matched_eval, matched_stats, seeds, gate"
  exit 1
fi

case "$STAGE" in
  pipeline_full)
    bash "$ROOT/scripts/run_pipeline_full.sh"
    ;;
  c5_aggressive)
    bash "$ROOT/scripts/run_c5_aggressive.sh"
    ;;
  c5r)
    bash "$ROOT/scripts/run_c5r_scale.sh"
    ;;
  matched_build)
    bash "$ROOT/scripts/run_matched_nonmember.sh"
    ;;
  matched_eval)
    bash "$ROOT/scripts/run_eval_matched.sh"
    ;;
  matched_stats)
    bash "$ROOT/scripts/run_compute_stats_matched.sh"
    ;;
  seeds)
    bash "$ROOT/scripts/run_seed_reps.sh"
    ;;
  gate)
    bash "$ROOT/scripts/run_gate.sh"
    ;;
  *)
    echo "Unknown stage: $STAGE"
    exit 1
    ;;
esac
