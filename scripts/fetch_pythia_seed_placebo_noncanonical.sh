#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STAGING_DIR_DEFAULT="$(cd "$SCRIPT_DIR/../.." && pwd)/exp/outputs/myriad_seed_placebo_pythia_17_19_20260505"
FAMILIES=pythia PYTHIA_SEEDS="17 19" STAGING_DIR="${STAGING_DIR:-$STAGING_DIR_DEFAULT}" "$SCRIPT_DIR/fetch_seed_placebo_c1_retain.sh" "$@"
