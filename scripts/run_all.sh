#!/usr/bin/env bash
set -euo pipefail

bash scripts/run_gate.sh
bash scripts/run_build.sh
bash scripts/run_train_teachers.sh
bash scripts/run_distill.sh
bash scripts/run_eval.sh
