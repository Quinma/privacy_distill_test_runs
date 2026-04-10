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
    if module -t avail python/3.10.4 2>&1 | grep -q "python/3.10.4"; then
      module load python/3.10.4
    elif module -t avail python/3.9.0 2>&1 | grep -q "python/3.9.0"; then
      module load python/3.9.0
    elif module -t avail python/3.8.0 2>&1 | grep -q "python/3.8.0"; then
      module load python/3.8.0
    fi
  fi
fi

# Example modules (adjust as needed)
# module load python/3.10.4
# module load cuda/12.1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
