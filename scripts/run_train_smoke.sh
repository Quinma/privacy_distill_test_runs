#!/usr/bin/env bash
set -euo pipefail

# Smoke test for bf16 + 8-bit AdamW + batch/accum on a tiny dataset.
# Uses GPU3 by default.
DATASET=${SMOKE_DATASET:-data/datasets/teacher_c2}
OUT=${SMOKE_OUT:-outputs/teachers/smoke}
BATCH=${BATCH:-2}
ACCUM=${ACCUM:-16}

if [[ ! -d "$DATASET" ]]; then
  echo "Smoke dataset not found: $DATASET" >&2
  exit 1
fi

python - <<'PY'
from datasets import load_from_disk
import os

dsrc = os.environ.get('SMOKE_DATASET', 'data/datasets/teacher_c2')
ddst = 'data/datasets/teacher_smoke'

from pathlib import Path
if not Path(dsrc).exists():
    raise SystemExit(f"Missing dataset: {dsrc}")

ds = load_from_disk(dsrc)
subset = ds.select(range(min(512, len(ds))))
if Path(ddst).exists():
    import shutil
    shutil.rmtree(ddst)
subset.save_to_disk(ddst)
print('smoke dataset size', len(subset))
PY

CUDA_VISIBLE_DEVICES=3 python -u src/train_teacher.py \
  --dataset data/datasets/teacher_smoke \
  --output "$OUT" \
  --per-device-batch "$BATCH" \
  --grad-accum "$ACCUM" \
  --epochs 1 \
  --bf16 \
  --optim adamw_8bit \
  --max-steps 10 \
  --dataloader-num-workers 2 \
  --dataloader-pin-memory
