#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-ucacmqu@myriad.rc.ucl.ac.uk}"
REMOTE_ROOT="${REMOTE_ROOT:-/myriadfs/home/ucacmqu/privacy_distill_test_runs}"
STAGING_DIR="${STAGING_DIR:-$WORKSPACE_ROOT/exp/outputs/myriad_placebo_c6_20260425}"

mkdir -p "$STAGING_DIR/outputs" "$STAGING_DIR/logs"

rsync -av --partial --append-verify \
  "$REMOTE_HOST:$REMOTE_ROOT/outputs/pythia-1.4b-placebo-npo-s13/" \
  "$STAGING_DIR/outputs/pythia-1.4b-placebo-npo-s13/" || true

rsync -av --partial --append-verify --include='*/' --include='*p14-placebo-c6*' --exclude='*' \
  "$REMOTE_HOST:$REMOTE_ROOT/logs/" \
  "$STAGING_DIR/logs/" || true

echo "Fetch complete: $STAGING_DIR"
