#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

BUILD="${BUILD:-0}"
SEEDS="${SEEDS:-13 17 19}"
RUN_PLACEBO="${RUN_PLACEBO:-1}"
RUN_FOLLOWUPS="${RUN_FOLLOWUPS:-1}"

if [[ "$BUILD" == "1" ]]; then
  bash scripts/run_gate.sh
  bash scripts/run_build.sh
fi

bash scripts/run_train_teachers.sh
bash scripts/run_distill.sh
bash scripts/run_eval.sh
bash scripts/run_c6_npo.sh
SEEDS="$SEEDS" bash scripts/run_seed_reps_1p4b.sh

if [[ "$RUN_PLACEBO" == "1" ]]; then
  bash scripts/run_placebo_c6_npo_1p4b.sh
  FAMILIES=pythia PYTHIA_SEEDS="$SEEDS" bash scripts/run_seed_placebo_c1_retain.sh
fi

if [[ "$RUN_FOLLOWUPS" == "1" ]]; then
  bash scripts/run_participation_followups.sh
fi

"$ROOT/.venv/bin/python" scripts/make_tables.py --family pythia-1.4b
