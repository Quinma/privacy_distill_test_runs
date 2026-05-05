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
REMOTE_TARGET="${REMOTE_TARGET:-$REMOTE_ROOT/data/datasets/gpt-neo-fixed-20260419/eval_target_holdout}"

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

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "REMOTE_ROOT='$REMOTE_ROOT' REMOTE_RETAIN='$REMOTE_RETAIN' REMOTE_TARGET='$REMOTE_TARGET' bash -s" <<'REMOTE'
set -euo pipefail

ROOT="${REMOTE_ROOT:-/path/to/privacy_distill_test_runs}"
RETAIN="${REMOTE_RETAIN:-$ROOT/data/datasets/eval_retain_holdout}"
TARGET="${REMOTE_TARGET:-$ROOT/data/datasets/gpt-neo-fixed-20260419/eval_target_holdout}"
cd "$ROOT"

RUN_CMD=$(cat <<'CMD'
set -euo pipefail

PY="$REPO_ROOT/.venv/bin/python"
TAG="gpt-neo-1.3b-local"
OUT="$REPO_ROOT/outputs/$TAG"
RETAIN="$REPO_ROOT/data/datasets/eval_retain_holdout"
TARGET="${TARGET_DATASET:-$REPO_ROOT/data/datasets/gpt-neo-fixed-20260419/eval_target_holdout}"
ATTACK_DIR="$OUT/mia_teacher_attack"
mkdir -p "$ATTACK_DIR"

if [[ ! -d "$RETAIN" ]]; then
  echo "ERROR: retain holdout dataset missing: $RETAIN" >&2
  exit 1
fi
if [[ ! -d "$TARGET" ]]; then
  echo "ERROR: target holdout dataset missing: $TARGET" >&2
  exit 1
fi

teacher_model() {
  local condition="$1"
  local path="$OUT/teachers/$condition"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: missing teacher model: $path" >&2
    exit 1
  fi
  printf '%s\n' "$path"
}

eval_teacher_pool() {
  local model="$1"
  local dataset="$2"
  local out_json="$3"
  "$PY" - "$model" "$dataset" "$out_json" <<'PY'
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
device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = int(os.environ.get("TEACHER_ATTACK_BATCH_SIZE", "8"))
max_length = int(os.environ.get("TEACHER_ATTACK_MAX_LENGTH", "512"))
max_chars = int(os.environ.get("TEACHER_ATTACK_MAX_CHARS", "32768"))

tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device == "cuda" else torch.float32)
model = AutoModelForCausalLM.from_pretrained(str(model_path)).to(device=device, dtype=dtype)
model.eval()

ds = datasets.load_from_disk(str(dataset_path))
items = [(str(ex.get("cik", "")), ex.get("text", "")) for ex in ds]
items = [(cik, text) for cik, text in items if cik and text]

grouped = {}
with torch.inference_mode():
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
        done = min(i + batch_size, len(items))
        if done == len(batch) or done % 40 == 0 or done == len(items):
            print(json.dumps({"model": str(model_path), "dataset": str(dataset_path), "done": done, "total": len(items)}), flush=True)

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
print(json.dumps({"output": str(output_path), "num_companies": len(grouped)}, indent=2), flush=True)
PY
}

build_attack() {
  local c6_target="$1"
  local c3_target="$2"
  local c1_target="$3"
  local c6_retain="$4"
  local c3_retain="$5"
  local c1_retain="$6"
  local out_c3="$7"
  local out_c1="$8"
  "$PY" - "$c6_target" "$c3_target" "$c1_target" "$c6_retain" "$c3_retain" "$c1_retain" "$out_c3" "$out_c1" <<'PY'
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
        "attack": "deletion_target_inference_teacher",
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
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


c6_target_path, c3_target_path, c1_target_path, c6_retain_path, c3_retain_path, c1_retain_path, out_c3_path, out_c1_path = map(Path, sys.argv[1:])
c6_target = load_company_losses(c6_target_path)
c3_target = load_company_losses(c3_target_path)
c1_target = load_company_losses(c1_target_path)
c6_retain = load_company_losses(c6_retain_path)
c3_retain = load_company_losses(c3_retain_path)
c1_retain = load_company_losses(c1_retain_path)

build(c6_target, c3_target, c6_retain, c3_retain, "C6_teacher_minus_C3_teacher", out_c3_path)
build(c6_target, c1_target, c6_retain, c1_retain, "C6_teacher_minus_C1_teacher", out_c1_path)
PY
}

C1_MODEL="$(teacher_model c1)"
C3_MODEL="$(teacher_model c3)"
C6_MODEL="$(teacher_model c6_unlearn)"

for cond in c1 c3 c6; do
  case "$cond" in
    c1) model="$C1_MODEL" ;;
    c3) model="$C3_MODEL" ;;
    c6) model="$C6_MODEL" ;;
  esac
  for split in target retain; do
    case "$split" in
      target) dataset="$TARGET" ;;
      retain) dataset="$RETAIN" ;;
    esac
    out_json="$ATTACK_DIR/${cond}_teacher_${split}.json"
    if [[ -f "$out_json" ]]; then
      echo "SKIP teacher eval ${cond}/${split}: $out_json exists"
      continue
    fi
    CUDA_VISIBLE_DEVICES=0 eval_teacher_pool "$model" "$dataset" "$out_json"
  done
done

build_attack \
  "$ATTACK_DIR/c6_teacher_target.json" \
  "$ATTACK_DIR/c3_teacher_target.json" \
  "$ATTACK_DIR/c1_teacher_target.json" \
  "$ATTACK_DIR/c6_teacher_retain.json" \
  "$ATTACK_DIR/c3_teacher_retain.json" \
  "$ATTACK_DIR/c1_teacher_retain.json" \
  "$OUT/mia_c6_teacher_deletion_attack_target_vs_retain_c3ref.json" \
  "$OUT/mia_c6_teacher_deletion_attack_target_vs_retain_c1ref.json"

echo "Wrote teacher target-vs-retain deletion attacks under $OUT"
CMD
)

qsub -N neo13-tdel \
  -l h_rt=06:00:00 -l mem=8G -l gpu=1 \
  -pe smp 4 -ac allow=L \
  -v REPO_ROOT="$ROOT",RUN_CMD="$RUN_CMD",TARGET_DATASET="$TARGET" \
  cluster/qsub_run.sh
REMOTE
