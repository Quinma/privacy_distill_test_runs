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

for ds in "$RETAIN" "$TARGET"; do
  if [[ ! -d "$ds" ]]; then
    echo "ERROR: dataset missing: $ds" >&2
    exit 1
  fi
done

for model_dir in "$OUT/teachers/c1" "$OUT/teachers/c3" "$OUT/teachers/c6_unlearn"; do
  if [[ ! -d "$model_dir" ]]; then
    echo "ERROR: missing teacher model: $model_dir" >&2
    exit 1
  fi
done

CUDA_VISIBLE_DEVICES=0 "$PY" - "$OUT" "$TARGET" "$RETAIN" <<'PY'
import json
import math
import os
import random
import sys
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


out_root = Path(sys.argv[1])
target_dataset = Path(sys.argv[2])
retain_dataset = Path(sys.argv[3])
c6_model_path = out_root / "teachers" / "c6_unlearn"
refs = ["c1", "c3"]
batch_size = int(os.environ.get("TEACHER_FEATURE_BATCH_SIZE", "2"))
max_length = int(os.environ.get("TEACHER_FEATURE_MAX_LENGTH", "512"))
max_chars = int(os.environ.get("TEACHER_FEATURE_MAX_CHARS", "32768"))
kl_top_fracs = [float(x) for x in os.environ.get("TEACHER_FEATURE_KL_TOP_FRACS", "0.10,0.01").split(",") if x]

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (torch.float16 if device == "cuda" else torch.float32)


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


def top_mean(values, frac):
    if not values:
        return float("nan")
    k = max(1, int(math.ceil(len(values) * frac)))
    arr = np.asarray(values, dtype=np.float64)
    idx = np.argpartition(arr, -k)[-k:]
    return float(arr[idx].mean())


def summarize_company_features(feature_lists):
    out = {}
    for cik, feats in feature_lists.items():
        row = {
            "num_tokens": len(feats["loss_delta"]),
            "mean_loss_delta": float(np.mean(feats["loss_delta"])),
            "token_kl_mean": float(np.mean(feats["token_kl"])),
            "gold_logit_diff_mean": float(np.mean(feats["gold_logit_diff"])),
        }
        for frac in kl_top_fracs:
            pct = int(round(frac * 100))
            row[f"token_kl_top{pct}_mean"] = top_mean(feats["token_kl"], frac)
        out[cik] = row
    return out


def score_dataset(c6_model, ref_model, tokenizer, dataset_path):
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
            labels = input_ids[:, 1:].contiguous()
            mask = attention_mask[:, 1:].contiguous().bool()

            c6_logits = c6_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :].float()
            ref_logits = ref_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :].float()

            c6_log_probs = F.log_softmax(c6_logits, dim=-1)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            c6_probs = c6_log_probs.exp()

            loss_c6 = F.nll_loss(
                c6_log_probs.reshape(-1, c6_log_probs.size(-1)),
                labels.reshape(-1),
                reduction="none",
            ).reshape(labels.shape)
            loss_ref = F.nll_loss(
                ref_log_probs.reshape(-1, ref_log_probs.size(-1)),
                labels.reshape(-1),
                reduction="none",
            ).reshape(labels.shape)
            loss_delta = loss_c6 - loss_ref

            token_kl = (c6_probs * (c6_log_probs - ref_log_probs)).sum(dim=-1)
            gold_logit_c6 = c6_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            gold_logit_ref = ref_logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            gold_logit_diff = gold_logit_c6 - gold_logit_ref

            for b_idx, cik in enumerate(ciks):
                active = mask[b_idx].detach().cpu().numpy().astype(bool)
                if not active.any():
                    continue
                row = grouped.setdefault(cik, {
                    "loss_delta": [],
                    "token_kl": [],
                    "gold_logit_diff": [],
                })
                row["loss_delta"].extend(loss_delta[b_idx].detach().cpu().numpy()[active].tolist())
                row["token_kl"].extend(token_kl[b_idx].detach().cpu().numpy()[active].tolist())
                row["gold_logit_diff"].extend(gold_logit_diff[b_idx].detach().cpu().numpy()[active].tolist())

            done = min(i + batch_size, len(items))
            if done == len(batch) or done % 40 == 0 or done == len(items):
                print(json.dumps({"dataset": str(dataset_path), "done": done, "total": len(items)}), flush=True)
    return summarize_company_features(grouped)


def build_attack_rows(reference_name, target_scores, retain_scores, file_path):
    rows = []
    common_features = sorted(set(next(iter(target_scores.values())).keys()) & set(next(iter(retain_scores.values())).keys()))
    for feature in common_features:
        if feature == "num_tokens":
            continue
        pos = [v[feature] for v in target_scores.values() if v.get(feature) is not None and not math.isnan(v[feature])]
        neg = [v[feature] for v in retain_scores.values() if v.get(feature) is not None and not math.isnan(v[feature])]
        if not pos or not neg:
            continue
        ci = bootstrap_auc(pos, neg)
        rows.append({
            "scale": "neo-1.3B",
            "reference": reference_name,
            "candidate_pool": "retained_companies",
            "positive_class": "deleted_targets",
            "negative_class": "retained_companies",
            "feature": feature,
            "auroc": auc(pos, neg),
            "ci_low": ci[0],
            "ci_high": ci[1],
            "positive_mean": float(np.mean(pos)),
            "negative_mean": float(np.mean(neg)),
            "n_positive": len(pos),
            "n_negative": len(neg),
            "file": str(file_path),
        })
    return rows


attack_dir = out_root / "mia_teacher_attack"
attack_dir.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(str(c6_model_path), use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

c6_model = AutoModelForCausalLM.from_pretrained(str(c6_model_path)).to(device=device, dtype=dtype)
c6_model.eval()

summary_rows = []
for ref in refs:
    ref_model_path = out_root / "teachers" / ref
    ref_model = AutoModelForCausalLM.from_pretrained(str(ref_model_path)).to(device=device, dtype=dtype)
    ref_model.eval()

    target_scores = score_dataset(c6_model, ref_model, tokenizer, target_dataset)
    retain_scores = score_dataset(c6_model, ref_model, tokenizer, retain_dataset)
    raw_path = attack_dir / f"teacher_feature_scores_{ref}.json"
    raw = {
        "reference": ref,
        "target_dataset": str(target_dataset),
        "retain_dataset": str(retain_dataset),
        "target_scores_by_company": target_scores,
        "retain_scores_by_company": retain_scores,
    }
    raw_path.write_text(json.dumps(raw, indent=2, sort_keys=True) + "\n")
    summary_rows.extend(build_attack_rows(ref, target_scores, retain_scores, raw_path))

    del ref_model
    if device == "cuda":
        torch.cuda.empty_cache()

summary = {
    "rows": summary_rows,
    "features": sorted({row["feature"] for row in summary_rows}),
    "references": refs,
}
summary_path = attack_dir / "teacher_feature_attack_summary.json"
summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print(f"Wrote {summary_path}")
PY

echo "Wrote teacher feature attacks under $ATTACK_DIR"
CMD
)

qsub -N neo13-tfeat \
  -l h_rt=08:00:00 -l mem=8G -l gpu=1 \
  -pe smp 4 -ac allow=L \
  -v REPO_ROOT="$ROOT",RUN_CMD="$RUN_CMD",TARGET_DATASET="$TARGET" \
  cluster/qsub_run.sh
REMOTE
