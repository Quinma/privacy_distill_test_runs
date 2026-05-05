#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"

BASE_RUN_TAG="${BASE_RUN_TAG:-pythia-1.4b}"
RUN_TAG="${RUN_TAG:-pythia-1.4b-placebo-npo-s13}"
MODEL="${MODEL:-EleutherAI/pythia-1.4b}"
STUDENT="${STUDENT:-EleutherAI/pythia-410m}"
SEED="${SEED:-13}"

BASE_OUT_ROOT="${BASE_OUT_ROOT:-$ROOT/outputs/$BASE_RUN_TAG}"
OUT_ROOT="${OUT_ROOT:-$ROOT/outputs/$RUN_TAG}"
DATASETS_DIR="${DATASETS_DIR:-$ROOT/data/datasets/$BASE_RUN_TAG}"
RETAIN_HOLDOUT_DATA="${RETAIN_HOLDOUT_DATA:-$ROOT/data/datasets/eval_retain_holdout}"

FORGET_DATA="${FORGET_DATA:-$DATASETS_DIR/random_forget_train}"
RETAIN_DATA="${RETAIN_DATA:-$DATASETS_DIR/teacher_c2}"
POLICY_INIT="${POLICY_INIT:-$BASE_OUT_ROOT/teachers/c1}"
REF_MODEL="${REF_MODEL:-$BASE_OUT_ROOT/teachers/c1}"
UNLEARN_OUT="${UNLEARN_OUT:-$OUT_ROOT/teachers/c6_placebo_unlearn}"
STUDENT_OUT="${STUDENT_OUT:-$OUT_ROOT/students/c6_placebo}"
MIA_DIR="${MIA_DIR:-$OUT_ROOT/mia}"
REF_EVAL_DIR="${REF_EVAL_DIR:-$OUT_ROOT/reference_eval}"
RETAIN_EVAL_DIR="${RETAIN_EVAL_DIR:-$OUT_ROOT/mia_retain}"

MAX_LENGTH="${MAX_LENGTH:-512}"
BF16="${BF16:-1}"
EVAL_GPU="${EVAL_GPU:-0}"

mkdir -p "$OUT_ROOT" "$MIA_DIR" "$REF_EVAL_DIR" "$RETAIN_EVAL_DIR"

bf16_flag() {
  if [[ "$BF16" == "1" ]]; then
    printf '%s' "--bf16"
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: missing directory: $path" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing file: $path" >&2
    exit 1
  fi
}

require_dir "$DATASETS_DIR/distill"
require_dir "$DATASETS_DIR/eval_target_holdout"
require_dir "$DATASETS_DIR/eval_nonmember"
require_dir "$RETAIN_HOLDOUT_DATA"
require_dir "$FORGET_DATA"
require_dir "$RETAIN_DATA"
require_file "$DATASETS_DIR/holdout_map.json"
require_dir "$POLICY_INIT"

export MODEL
export STUDENT
export OUT_ROOT
export DATASETS_DIR
export POLICY_INIT
export REF_MODEL
export FORGET_DATA
export RETAIN_DATA
export UNLEARN_OUT
export STUDENT_OUT
export MIA_DIR
export SEED
export BF16
export EVAL_GPU

bash "$ROOT/scripts/run_c6_npo.sh"

for ref in c1 c3; do
  ref_model="$BASE_OUT_ROOT/students/$ref"
  ref_target_json="$REF_EVAL_DIR/${ref}_student_target.json"
  ref_retain_json="$REF_EVAL_DIR/${ref}_student_retain.json"

  if [[ ! -f "$ref_target_json" ]]; then
    require_dir "$ref_model"
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PY" "$ROOT/src/eval_mia.py" \
      --model "$ref_model" \
      --target-holdout "$DATASETS_DIR/eval_target_holdout" \
      --nonmember "$DATASETS_DIR/eval_nonmember" \
      --holdout-map "$DATASETS_DIR/holdout_map.json" \
      --output "$ref_target_json" \
      --batch-size 4 \
      --max-length "$MAX_LENGTH" \
      $(bf16_flag)
  fi

  if [[ ! -f "$ref_retain_json" ]]; then
    require_dir "$ref_model"
    CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PY" - "$ref_model" "$RETAIN_HOLDOUT_DATA" "$ref_retain_json" "$(bf16_flag)" <<'PY'
import json
import sys
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


model_path = Path(sys.argv[1])
dataset_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
use_bf16 = len(sys.argv) > 4 and sys.argv[4] == "--bf16"
batch_size = 8
max_length = 512
max_chars = 32768

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if use_bf16 and device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device == "cuda" else torch.float32)

tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(str(model_path)).to(device=device, dtype=dtype if device == "cuda" else torch.float32)
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
        enc = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
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

result = {
    "dataset": str(dataset_path),
    "model": str(model_path),
    "num_companies": len(grouped),
    "per_company": {
        cik: {"num_docs": len(losses), "mean_loss": float(np.mean(losses))}
        for cik, losses in sorted(grouped.items())
    },
}
output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
PY
  fi
done

PLACEBO_TARGET_JSON="$MIA_DIR/c6_student.json"
PLACEBO_RETAIN_JSON="$RETAIN_EVAL_DIR/c6_placebo_student_retain.json"

if [[ ! -f "$PLACEBO_RETAIN_JSON" ]]; then
  CUDA_VISIBLE_DEVICES="$EVAL_GPU" "$PY" - "$STUDENT_OUT" "$RETAIN_HOLDOUT_DATA" "$PLACEBO_RETAIN_JSON" "$(bf16_flag)" <<'PY'
import json
import sys
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


model_path = Path(sys.argv[1])
dataset_path = Path(sys.argv[2])
output_path = Path(sys.argv[3])
use_bf16 = len(sys.argv) > 4 and sys.argv[4] == "--bf16"
batch_size = 8
max_length = 512
max_chars = 32768

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if use_bf16 and device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device == "cuda" else torch.float32)

tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(str(model_path)).to(device=device, dtype=dtype if device == "cuda" else torch.float32)
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
        enc = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
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

result = {
    "dataset": str(dataset_path),
    "model": str(model_path),
    "num_companies": len(grouped),
    "per_company": {
        cik: {"num_docs": len(losses), "mean_loss": float(np.mean(losses))}
        for cik, losses in sorted(grouped.items())
    },
}
output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
PY
fi

"$PY" - \
  "$PLACEBO_TARGET_JSON" \
  "$REF_EVAL_DIR/c3_student_target.json" \
  "$REF_EVAL_DIR/c1_student_target.json" \
  "$PLACEBO_RETAIN_JSON" \
  "$REF_EVAL_DIR/c3_student_retain.json" \
  "$REF_EVAL_DIR/c1_student_retain.json" \
  "$OUT_ROOT/mia_c6_placebo_deletion_attack_target_vs_retain.json" \
  "$OUT_ROOT/mia_c6_placebo_deletion_attack_target_vs_retain_c1ref.json" \
  "$SEED" <<'PY'
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


def build(placebo_target, ref_target, placebo_retain, ref_retain, reference, out_path, seed):
    target_keys = sorted(set(placebo_target) & set(ref_target))
    retain_keys = sorted(set(placebo_retain) & set(ref_retain))
    target_delta = {k: placebo_target[k] - ref_target[k] for k in target_keys}
    retain_delta = {k: placebo_retain[k] - ref_retain[k] for k in retain_keys}
    target_vals = list(target_delta.values())
    retain_vals = list(retain_delta.values())
    retain_mean = float(np.mean(retain_vals))
    result = {
        "seed": int(seed),
        "control": "disjoint_wrong_target_placebo_npo",
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


placebo_target_path, c3_target_path, c1_target_path, placebo_retain_path, c3_retain_path, c1_retain_path, out_c3_path, out_c1_path = map(Path, sys.argv[1:9])
seed = int(sys.argv[9])
placebo_target = load_company_losses(placebo_target_path)
c3_target = load_company_losses(c3_target_path)
c1_target = load_company_losses(c1_target_path)
placebo_retain = load_company_losses(placebo_retain_path)
c3_retain = load_company_losses(c3_retain_path)
c1_retain = load_company_losses(c1_retain_path)

build(placebo_target, c3_target, placebo_retain, c3_retain, "C3", out_c3_path, seed)
build(placebo_target, c1_target, placebo_retain, c1_retain, "C1", out_c1_path, seed)
PY

echo "Placebo C6 NPO control complete under $OUT_ROOT"
