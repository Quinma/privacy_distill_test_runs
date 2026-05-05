#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-ucacmqu@myriad.rc.ucl.ac.uk}"
REMOTE_ROOT="${REMOTE_ROOT:-/myriadfs/home/ucacmqu/privacy_distill_test_runs}"
CONTROL_PATH="${CONTROL_PATH:-$HOME/.ssh/cm-%r@%h:%p}"
SSH_OPTS=(-o ControlMaster=auto -o ControlPersist=15m -o ControlPath="$CONTROL_PATH")

mkdir -p "$(dirname "$CONTROL_PATH")"

cleanup() {
  ssh "${SSH_OPTS[@]}" -O exit "$REMOTE_HOST" >/dev/null 2>&1 || true
}
trap cleanup EXIT

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "REMOTE_ROOT='$REMOTE_ROOT' bash -s" <<'REMOTE'
set -euo pipefail

ROOT="${REMOTE_ROOT:-/myriadfs/home/ucacmqu/privacy_distill_test_runs}"
cd "$ROOT"

RUN_CMD=$(cat <<'CMD'
set -euo pipefail

PY="$REPO_ROOT/.venv/bin/python"
TAG="gpt-neo-2.7b"
DATA="$REPO_ROOT/data/datasets/gpt-neo-fixed-20260419"
OUT="$REPO_ROOT/outputs/$TAG"
MIA="$OUT/mia"

mkdir -p "$MIA"

CUDA_VISIBLE_DEVICES=0 "$PY" src/eval_ppl.py \
  --model "$OUT/students/c1" \
  --dataset "$DATA/distill" \
  --output "$MIA/utility_c1_student.json" \
  --batch-size 4 --max-samples 500 --bf16

CUDA_VISIBLE_DEVICES=0 "$PY" src/eval_ppl.py \
  --model "$OUT/students/c2" \
  --dataset "$DATA/distill" \
  --output "$MIA/utility_c2_student.json" \
  --batch-size 4 --max-samples 500 --bf16

CUDA_VISIBLE_DEVICES=0 "$PY" src/eval_ppl.py \
  --model "$OUT/students/c1" \
  --dataset "$DATA/eval_target_holdout" \
  --output "$MIA/utility_c1_student_holdout.json" \
  --batch-size 4 --max-samples 250 --bf16

CUDA_VISIBLE_DEVICES=0 "$PY" src/eval_ppl.py \
  --model "$OUT/students/c2" \
  --dataset "$DATA/eval_target_holdout" \
  --output "$MIA/utility_c2_student_holdout.json" \
  --batch-size 4 --max-samples 250 --bf16

CUDA_VISIBLE_DEVICES=0 "$PY" src/eval_mia.py \
  --model "$OUT/teachers/c2" \
  --target-holdout "$DATA/eval_target_holdout" \
  --nonmember "$DATA/eval_nonmember" \
  --holdout-map "$DATA/holdout_map.json" \
  --output "$MIA/c2_teacher.json" \
  --batch-size 4 --max-length 512 --bf16

CUDA_VISIBLE_DEVICES=0 "$PY" src/eval_ppl.py \
  --model "$OUT/teachers/c2" \
  --dataset "$DATA/distill" \
  --output "$MIA/utility_c2_teacher.json" \
  --batch-size 4 --max-samples 500 --bf16

"$PY" - "$MIA/c4_teacher.json" "$MIA/c2_teacher.json" "$MIA/teacher_c1_vs_c2.json" <<'PY'
import json
import random
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

c1_path, c2_path, out_path = map(Path, sys.argv[1:4])
c1 = json.loads(c1_path.read_text())["per_company"]
c2 = json.loads(c2_path.read_text())["per_company"]
common = sorted(set(c1) & set(c2))
a = [c1[k]["mean_loss"] for k in common]
b = [c2[k]["mean_loss"] for k in common]
diffs = [x - y for x, y in zip(a, b)]

pos = sum(1 for d in diffs if d > 0)
neg = sum(1 for d in diffs if d < 0)
n_sign = pos + neg
sign_p = float(binomtest(pos, n_sign, 0.5, alternative="two-sided").pvalue) if n_sign else 1.0

rng = random.Random(13)
obs = float(np.mean(diffs)) if diffs else None
count = 0
perm_n = 20000
for _ in range(perm_n):
    flipped = [d if rng.random() < 0.5 else -d for d in diffs]
    if abs(float(np.mean(flipped))) >= abs(obs):
        count += 1
perm_p = (count + 1) / (perm_n + 1)

rng = random.Random(13)
means = []
for _ in range(2000):
    sample = [rng.choice(diffs) for _ in diffs]
    means.append(float(np.mean(sample)))

out = {
    "comparison": "teacher_C1_vs_C2",
    "c1_file": str(c1_path),
    "c2_file": str(c2_path),
    "n": len(common),
    "mean_diff": obs,
    "bootstrap_ci_diff": [
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    ],
    "sign_test": {
        "pos": pos,
        "neg": neg,
        "n": n_sign,
        "p_value_two_sided": sign_p,
    },
    "perm_test": {
        "obs_mean_diff": obs,
        "p_value_two_sided": perm_p,
        "n": len(common),
    },
}
out_path.write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
PY
CMD
)

qsub -N neo27b-diag \
  -l h_rt=06:00:00 -l mem=8G -l gpu=1 \
  -pe smp 4 -ac allow=L \
  -v REPO_ROOT="$ROOT",RUN_CMD="$RUN_CMD" \
  cluster/qsub_run.sh
REMOTE
