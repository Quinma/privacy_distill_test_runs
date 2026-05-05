#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
FAMILIES=pythia PYTHIA_SEEDS="17 19" "$SCRIPT_DIR/sync_and_submit_seed_placebo_c1_retain.sh" "$@"
