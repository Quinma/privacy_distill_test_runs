#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-user@cluster.example.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/path/to/privacy_distill_test_runs}"
CONTROL_PATH="${CONTROL_PATH:-$HOME/.ssh/cm-%r@%h:%p}"
SSH_OPTS=(-o ControlMaster=auto -o ControlPersist=15m -o ControlPath="$CONTROL_PATH")
RSYNC_RSH="ssh ${SSH_OPTS[*]}"
RSYNC_OPTS=(-av --partial --append-verify)

LOCAL_RETAIN="${LOCAL_RETAIN:-$WORKSPACE_ROOT/exp/data/datasets/eval_retain_holdout}"
REMOTE_RETAIN="${REMOTE_RETAIN:-$REMOTE_ROOT/data/datasets/eval_retain_holdout}"

mkdir -p "$(dirname "$CONTROL_PATH")"

cleanup() {
  ssh "${SSH_OPTS[@]}" -O exit "$REMOTE_HOST" >/dev/null 2>&1 || true
}
trap cleanup EXIT

if [[ ! -d "$LOCAL_RETAIN" ]]; then
  echo "ERROR: missing local retain holdout dataset: $LOCAL_RETAIN" >&2
  exit 1
fi

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '$REMOTE_RETAIN'"
rsync "${RSYNC_OPTS[@]}" -e "$RSYNC_RSH" "$LOCAL_RETAIN/" "$REMOTE_HOST:$REMOTE_RETAIN/"

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "REMOTE_ROOT='$REMOTE_ROOT' REMOTE_RETAIN='$REMOTE_RETAIN' bash -s" <<'REMOTE'
set -euo pipefail

ROOT="${REMOTE_ROOT:-/path/to/privacy_distill_test_runs}"
RETAIN="${REMOTE_RETAIN:-$ROOT/data/datasets/eval_retain_holdout}"
cd "$ROOT"

RUN_CMD=$(cat <<'CMD'
set -euo pipefail

PY="$REPO_ROOT/.venv/bin/python"
TAG="gpt-neo-1.3b-local"
OUT="$REPO_ROOT/outputs/$TAG"
DATA="$REPO_ROOT/data/datasets/$TAG"
RETAIN="$REPO_ROOT/data/datasets/eval_retain_holdout"
NONMEMBER="$DATA/eval_nonmember"
HOLDOUT_MAP="$DATA/holdout_map.json"

if [[ ! -d "$RETAIN" ]]; then
  echo "ERROR: retain holdout dataset missing: $RETAIN" >&2
  exit 1
fi

RUN_BASE="$OUT"
if [[ -d "$OUT/seed_reps/seed_13/students/c6" ]]; then
  RUN_BASE="$OUT/seed_reps/seed_13"
fi

model_path() {
  local condition="$1"
  if [[ -d "$RUN_BASE/students/$condition" ]]; then
    printf '%s\n' "$RUN_BASE/students/$condition"
  elif [[ -d "$OUT/students/$condition" ]]; then
    printf '%s\n' "$OUT/students/$condition"
  else
    echo "ERROR: missing model for $condition under $RUN_BASE or $OUT" >&2
    exit 1
  fi
}

target_json() {
  local condition="$1"
  if [[ -f "$RUN_BASE/mia/${condition}_student.json" ]]; then
    printf '%s\n' "$RUN_BASE/mia/${condition}_student.json"
  elif [[ -f "$OUT/mia/${condition}_student.json" ]]; then
    printf '%s\n' "$OUT/mia/${condition}_student.json"
  else
    echo "ERROR: missing target MIA JSON for $condition under $RUN_BASE or $OUT" >&2
    exit 1
  fi
}

C1_MODEL="$(model_path c1)"
C3_MODEL="$(model_path c3)"
C6_MODEL="$(model_path c6)"
C1_TARGET="$(target_json c1)"
C3_TARGET="$(target_json c3)"
C6_TARGET="$(target_json c6)"
RETAIN_MIA="$RUN_BASE/mia_retain"
mkdir -p "$RETAIN_MIA"

echo "RUN_BASE=$RUN_BASE"
echo "C1_MODEL=$C1_MODEL"
echo "C3_MODEL=$C3_MODEL"
echo "C6_MODEL=$C6_MODEL"
echo "C1_TARGET=$C1_TARGET"
echo "C3_TARGET=$C3_TARGET"
echo "C6_TARGET=$C6_TARGET"

eval_retain_only() {
  local model="$1"
  local out_json="$2"
  "$PY" - "$model" "$RETAIN" "$out_json" <<'PY'
import json
import os
import sys
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


model_path, dataset_path, output_path = map(Path, sys.argv[1:4])
device = "cuda"
batch_size = 8
max_length = 512
# SEC filings can be multi-megabyte strings. We only score the first
# max_length tokens, so cap raw text before tokenization to avoid spending
# hours tokenizing content that will be discarded.
max_chars = int(os.environ.get("RETAIN_EVAL_MAX_CHARS", "32768"))

tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(str(model_path)).to(dtype=torch.bfloat16).to(device)
model.eval()

ds = datasets.load_from_disk(str(dataset_path))
items = [(str(ex.get("cik", "")), ex.get("text", "")) for ex in ds]
items = [(cik, text) for cik, text in items if cik and text]

grouped = {}
with torch.no_grad():
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        ciks = [cik for cik, _ in batch]
        texts = [text[:max_chars] for _, text in batch]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()
        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.size())
        loss = (loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
        for cik, value in zip(ciks, loss.detach().cpu().tolist()):
            grouped.setdefault(cik, []).append(float(value))
        if i == 0 or (i // batch_size + 1) % 10 == 0 or i + batch_size >= len(items):
            print(json.dumps({"model": str(model_path), "done": min(i + batch_size, len(items)), "total": len(items)}), flush=True)

result = {
    "dataset": str(dataset_path),
    "model": str(model_path),
    "num_companies": len(grouped),
    "per_company": {
        cik: {"num_docs": len(losses), "mean_loss": float(np.mean(losses))}
        for cik, losses in sorted(grouped.items())
    },
}
os.makedirs(output_path.parent, exist_ok=True)
output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps({"output": str(output_path), "num_companies": len(grouped)}, indent=2))
PY
}

for condition in c1 c3 c6; do
  model_var="${condition^^}_MODEL"
  model="${!model_var}"
  out_json="$RETAIN_MIA/${condition}_student_retain.json"
  if [[ -f "$out_json" ]]; then
    echo "SKIP retain eval $condition: $out_json exists"
    continue
  fi
  RETAIN_EVAL_MAX_CHARS=32768 CUDA_VISIBLE_DEVICES=0 eval_retain_only "$model" "$out_json"
done

"$PY" - "$C6_TARGET" "$C3_TARGET" "$C1_TARGET" \
  "$RETAIN_MIA/c6_student_retain.json" \
  "$RETAIN_MIA/c3_student_retain.json" \
  "$RETAIN_MIA/c1_student_retain.json" \
  "$RUN_BASE/mia_c6_deletion_attack_target_vs_retain.json" \
  "$RUN_BASE/mia_c6_deletion_attack_target_vs_retain_c1ref.json" <<'PY'
import json
import random
import sys
from pathlib import Path

import numpy as np


def load_company_losses(path: Path):
    data = json.loads(path.read_text())
    return {
        str(k): float(v["mean_loss"])
        for k, v in data["per_company"].items()
        if v.get("mean_loss") is not None
    }


def auc(pos, neg):
    scores = [(x, 1) for x in pos] + [(x, 0) for x in neg]
    scores.sort(key=lambda item: item[0])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(scores):
        j = i + 1
        while j < len(scores) and scores[j][0] == scores[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    n_pos = len(pos)
    n_neg = len(neg)
    rank_sum_pos = sum(rank for rank, (_, label) in zip(ranks, scores) if label == 1)
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def bootstrap_auc(pos, neg, seed=13, n_boot=2000):
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        pos_sample = [rng.choice(pos) for _ in pos]
        neg_sample = [rng.choice(neg) for _ in neg]
        vals.append(auc(pos_sample, neg_sample))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def build(c6_target, ref_target, c6_retain, ref_retain, reference, out_path):
    target_keys = sorted(set(c6_target) & set(ref_target))
    retain_keys = sorted(set(c6_retain) & set(ref_retain))
    target_delta = {k: c6_target[k] - ref_target[k] for k in target_keys}
    retain_delta = {k: c6_retain[k] - ref_retain[k] for k in retain_keys}
    target_vals = list(target_delta.values())
    retain_vals = list(retain_delta.values())
    retain_mean = float(np.mean(retain_vals))
    result = {
        "attack": "deletion_target_inference",
        "reference": reference,
        "positive_class": "deleted_targets",
        "negative_class": "retained_companies",
        "num_targets": len(target_vals),
        "num_retained": len(retain_vals),
        "auroc": auc(target_vals, retain_vals),
        "bootstrap_ci_95": bootstrap_auc(target_vals, retain_vals),
        "target_mean_delta": float(np.mean(target_vals)),
        "retain_mean_delta": retain_mean,
        "target_median_delta": float(np.median(target_vals)),
        "retain_median_delta": float(np.median(retain_vals)),
        "num_target_gt_retain_mean": int(sum(1 for x in target_vals if x > retain_mean)),
        "target_delta_by_company": target_delta,
        "retain_delta_by_company": retain_delta,
    }
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


c6_target_path, c3_target_path, c1_target_path, c6_retain_path, c3_retain_path, c1_retain_path, out_c3_path, out_c1_path = map(Path, sys.argv[1:])
c6_target = load_company_losses(c6_target_path)
c3_target = load_company_losses(c3_target_path)
c1_target = load_company_losses(c1_target_path)
c6_retain = load_company_losses(c6_retain_path)
c3_retain = load_company_losses(c3_retain_path)
c1_retain = load_company_losses(c1_retain_path)

build(c6_target, c3_target, c6_retain, c3_retain, "C6_minus_C3", out_c3_path)
build(c6_target, c1_target, c6_retain, c1_retain, "C6_minus_C1", out_c1_path)
PY

cp "$RUN_BASE/mia_c6_deletion_attack_target_vs_retain.json" "$OUT/mia_c6_deletion_attack_target_vs_retain.json"
cp "$RUN_BASE/mia_c6_deletion_attack_target_vs_retain_c1ref.json" "$OUT/mia_c6_deletion_attack_target_vs_retain_c1ref.json"
echo "Wrote target-vs-retain deletion attacks under $RUN_BASE and copied to $OUT"
CMD
)

qsub -N neo13-c6del \
  -l h_rt=04:00:00 -l mem=8G -l gpu=1 \
  -pe smp 4 -ac allow=L \
  -v REPO_ROOT="$ROOT",RUN_CMD="$RUN_CMD" \
  cluster/qsub_run.sh
REMOTE
