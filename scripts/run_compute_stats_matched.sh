#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"

MIA_14="$ROOT/outputs/pythia-1.4b/mia_matched"
MIA_28="$ROOT/outputs/pythia-2.8b/mia_matched"

run_stats() {
  local mia_dir="$1"
  local c1="$mia_dir/c1_student.json"
  local c2="$mia_dir/c2_student.json"
  local c3="$mia_dir/c3_student.json"
  local c4="$mia_dir/c4_teacher.json"
  local c5="$mia_dir/c5_student.json"

  if [[ ! -f "$c1" || ! -f "$c2" || ! -f "$c3" || ! -f "$c4" ]]; then
    echo "[run_compute_stats_matched] missing core files in $mia_dir"
    return 0
  fi

  if [[ -f "$c5" ]]; then
    $PY "$ROOT/src/compute_stats.py" --c1 "$c1" --c2 "$c2" --c3 "$c3" --c4 "$c4" --c5 "$c5" --out-dir "$mia_dir"
  else
    $PY "$ROOT/src/compute_stats.py" --c1 "$c1" --c2 "$c2" --c3 "$c3" --c4 "$c4" --out-dir "$mia_dir"
  fi
}

run_stats "$MIA_14"
run_stats "$MIA_28"

echo "[run_compute_stats_matched] done"
