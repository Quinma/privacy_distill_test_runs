#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-ucacmqu@myriad.rc.ucl.ac.uk}"
REMOTE_ROOT="${REMOTE_ROOT:-/myriadfs/home/ucacmqu/privacy_distill_test_runs}"
RUN_TAG="${RUN_TAG:-gpt-neo-1.3b-local-placebo-npo-s13}"
STAGING_DIR="${STAGING_DIR:-$WORKSPACE_ROOT/exp/outputs/myriad_placebo_neo13_20260426}"

mkdir -p "$STAGING_DIR/outputs" "$STAGING_DIR/logs"

rsync -av --partial --append-verify \
  "$REMOTE_HOST:$REMOTE_ROOT/outputs/$RUN_TAG/" \
  "$STAGING_DIR/outputs/$RUN_TAG/" || true

rsync -av --partial --append-verify --include='*/' --include='*neo13-placebo-c6*' --exclude='*' \
  "$REMOTE_HOST:$REMOTE_ROOT/logs/" \
  "$STAGING_DIR/logs/" || true

echo "Fetch complete: $STAGING_DIR"
