#!/usr/bin/env bash
set -euo pipefail

# Edit these for your cluster environment.
module purge
module load ucl-stack/2025-12
module load default-modules/2025-12

# Example modules (adjust as needed)
# module load python/3.10.4
# module load cuda/12.1

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
