#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "[run_matched_pipeline] building matched nonmember pool..."
bash "$ROOT/scripts/run_matched_nonmember.sh"

echo "[run_matched_pipeline] running matched evals..."
bash "$ROOT/scripts/run_eval_matched.sh"

echo "[run_matched_pipeline] computing matched stats..."
bash "$ROOT/scripts/run_compute_stats_matched.sh"

echo "[run_matched_pipeline] done"
