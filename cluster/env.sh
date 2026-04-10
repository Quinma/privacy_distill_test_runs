#!/usr/bin/env bash
set -euo pipefail

# Edit these for your cluster environment.
if command -v module >/dev/null 2>&1; then
  module purge
  if module -t avail ucl-stack/2025-12 2>&1 | grep -q "ucl-stack/2025-12"; then
    module load ucl-stack/2025-12
    module load default-modules/2025-12
  else
    module load default-modules/2018
    py_mod="$(module -t avail python 2>&1 | sed 's#.*/##' | grep -E '^python/[0-9]+' | sort -V | tail -n 1 || true)"
    if [[ -n "$py_mod" ]]; then
      module load "$py_mod"
    fi
  fi
fi

# Example modules (adjust as needed)
# module load python/3.10.4
# module load cuda/12.1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
