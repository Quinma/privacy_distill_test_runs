#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

REMOTE_HOST="${REMOTE_HOST:-ucacmqu@myriad.rc.ucl.ac.uk}"
REMOTE_ROOT="${REMOTE_ROOT:-/myriadfs/home/ucacmqu/privacy_distill_test_runs}"
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

ROOT="${REMOTE_ROOT:-/myriadfs/home/ucacmqu/privacy_distill_test_runs}"
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

for model_dir in "$OUT/teachers/c1" "$OUT/teachers/c3" "$OUT/teachers/c6_unlearn"; do
  if [[ ! -d "$model_dir" ]]; then
    echo "ERROR: missing teacher model: $model_dir" >&2
    exit 1
  fi
done

CUDA_VISIBLE_DEVICES=0 "$PY" - "$OUT/teachers/c6_unlearn" "$OUT/teachers/c1" "$OUT/teachers/c3" "$TARGET" "$RETAIN" "$ATTACK_DIR/teacher_kl_targets_retains.json" <<'PY'
import json
import math
import os
import sys
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


c6_model_path = Path(sys.argv[1])
c1_model_path = Path(sys.argv[2])
c3_model_path = Path(sys.argv[3])
target_dataset = Path(sys.argv[4])
retain_dataset = Path(sys.argv[5])
out_path = Path(sys.argv[6])

device = "cuda" if torch.cuda.is_available() else "cpu"
batch_size = int(os.environ.get("TEACHER_KL_BATCH_SIZE", "2"))
max_length = int(os.environ.get("TEACHER_KL_MAX_LENGTH", "512"))
max_chars = int(os.environ.get("TEACHER_KL_MAX_CHARS", "32768"))
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device == "cuda" else torch.float32)


def stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "n": int(arr.size),
    }


def summarize_pair(c6_model, ref_model, tokenizer, dataset_path):
    ds = datasets.load_from_disk(str(dataset_path))
    items = [(str(ex.get("cik", "")), ex.get("text", "")) for ex in ds]
    items = [(cik, text) for cik, text in items if cik and text]

    kl_c6_to_ref_vals = []
    kl_ref_to_c6_vals = []
    js_vals = []

    with torch.inference_mode():
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
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
            mask = attention_mask[:, 1:].contiguous().bool()

            c6_logits = c6_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :].float()
            ref_logits = ref_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :].float()

            c6_log_probs = F.log_softmax(c6_logits, dim=-1)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            c6_probs = c6_log_probs.exp()
            ref_probs = ref_log_probs.exp()
            m_probs = 0.5 * (c6_probs + ref_probs)
            m_log_probs = torch.log(m_probs.clamp_min(1e-12))

            kl_c6_to_ref = (c6_probs * (c6_log_probs - ref_log_probs)).sum(dim=-1)
            kl_ref_to_c6 = (ref_probs * (ref_log_probs - c6_log_probs)).sum(dim=-1)
            js = 0.5 * (
                (c6_probs * (c6_log_probs - m_log_probs)).sum(dim=-1) +
                (ref_probs * (ref_log_probs - m_log_probs)).sum(dim=-1)
            )

            kl_c6_to_ref_vals.extend(kl_c6_to_ref[mask].detach().cpu().tolist())
            kl_ref_to_c6_vals.extend(kl_ref_to_c6[mask].detach().cpu().tolist())
            js_vals.extend(js[mask].detach().cpu().tolist())

            done = min(i + batch_size, len(items))
            if done == len(batch) or done % 40 == 0 or done == len(items):
                print(json.dumps({"dataset": str(dataset_path), "done": done, "total": len(items)}), flush=True)

    return {
        "kl_c6_to_ref": stats(kl_c6_to_ref_vals),
        "kl_ref_to_c6": stats(kl_ref_to_c6_vals),
        "js_proxy_mean": float(np.mean(np.asarray(js_vals, dtype=np.float64))),
    }


tokenizer = AutoTokenizer.from_pretrained(str(c6_model_path), use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

c6_model = AutoModelForCausalLM.from_pretrained(str(c6_model_path)).to(device=device, dtype=dtype)
c6_model.eval()

result = {}
for ref_key, ref_path in [("c1_teacher", c1_model_path), ("c3_teacher", c3_model_path)]:
    ref_model = AutoModelForCausalLM.from_pretrained(str(ref_path)).to(device=device, dtype=dtype)
    ref_model.eval()
    result[ref_key] = {
        "target": summarize_pair(c6_model, ref_model, tokenizer, target_dataset),
        "retain": summarize_pair(c6_model, ref_model, tokenizer, retain_dataset),
    }
    del ref_model
    if device == "cuda":
        torch.cuda.empty_cache()

out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(f"Wrote {out_path}")
PY

echo "Wrote teacher KL diagnostics under $ATTACK_DIR"
CMD
)

qsub -N neo13-tkl \
  -l h_rt=06:00:00 -l mem=8G -l gpu=1 \
  -pe smp 4 -ac allow=L \
  -v REPO_ROOT="$ROOT",RUN_CMD="$RUN_CMD",TARGET_DATASET="$TARGET" \
  cluster/qsub_run.sh
REMOTE
