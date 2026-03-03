import argparse
import json
import os
from typing import Dict, List, Tuple

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import read_json, write_json


def _compute_losses(model, tokenizer, texts: List[str], batch_size: int, max_length: int, device: str) -> List[float]:
    losses = []
    model.eval()
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.size())
        loss = (loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
        losses.extend(loss.detach().cpu().tolist())
    return losses


def _compute_losses_by_key(
    model,
    tokenizer,
    items: List[Tuple[str, str]],
    batch_size: int,
    max_length: int,
    device: str,
) -> Tuple[Dict[str, List[float]], List[float]]:
    """Compute per-item losses and group them by key."""
    grouped: Dict[str, List[float]] = {}
    all_losses: List[float] = []
    model.eval()
    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        keys = [k for k, _ in batch]
        texts = [t for _, t in batch]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.size())
        loss = (loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
        loss_list = loss.detach().cpu().tolist()
        for k, l in zip(keys, loss_list):
            grouped.setdefault(k, []).append(l)
            all_losses.append(l)
    return grouped, all_losses


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--target-holdout", required=True, help="Dataset saved by data_prep")
    p.add_argument("--nonmember", required=True, help="Dataset saved by data_prep")
    p.add_argument("--holdout-map", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model)

    if args.bf16:
        model = model.to(dtype=torch.bfloat16)
    if args.fp16:
        model = model.to(dtype=torch.float16)
    model = model.to(args.device)

    holdout_map = read_json(args.holdout_map)

    target_ds = datasets.load_from_disk(args.target_holdout)
    nonmember_ds = datasets.load_from_disk(args.nonmember)

    # group target texts by cik
    target_texts_by_cik: Dict[str, List[str]] = {}
    for ex in target_ds:
        cik = str(ex.get("cik", ""))
        text = ex.get("text", "")
        if not cik:
            continue
        target_texts_by_cik.setdefault(cik, []).append(text)

    nonmember_items = []
    for ex in nonmember_ds:
        cik = str(ex.get("cik", ""))
        text = ex.get("text", "")
        if not cik or not text:
            continue
        nonmember_items.append((cik, text))

    nonmember_losses_by_cik, nonmember_losses = _compute_losses_by_key(
        model, tokenizer, nonmember_items, args.batch_size, args.max_length, args.device
    )
    nonmember_mean = float(np.mean(nonmember_losses)) if nonmember_losses else None
    nonmember_company_means = {
        cik: float(np.mean(losses)) for cik, losses in nonmember_losses_by_cik.items() if losses
    }

    per_company = {}
    member_losses_all = []

    for cik, texts in target_texts_by_cik.items():
        if not texts:
            continue
        losses = _compute_losses(
            model, tokenizer, texts, args.batch_size, args.max_length, args.device
        )
        member_losses_all.extend(losses)
        per_company[cik] = {
            "num_docs": len(texts),
            "mean_loss": float(np.mean(losses)) if losses else None,
        }
        if nonmember_mean is not None and per_company[cik]["mean_loss"] is not None:
            per_company[cik]["loss_ratio"] = per_company[cik]["mean_loss"] / nonmember_mean

    # AUROC (document-level): lower loss => member. Use negative loss as score.
    y_true_doc = [1] * len(member_losses_all) + [0] * len(nonmember_losses)
    y_scores_doc = [-l for l in member_losses_all] + [-l for l in nonmember_losses]
    auroc_doc = roc_auc_score(y_true_doc, y_scores_doc) if len(set(y_true_doc)) > 1 else None

    # AUROC (company-level): compare per-company mean losses.
    member_company_means = [
        v["mean_loss"] for v in per_company.values() if v.get("mean_loss") is not None
    ]
    nonmember_company_mean_list = list(nonmember_company_means.values())
    y_true_company = [1] * len(member_company_means) + [0] * len(nonmember_company_mean_list)
    y_scores_company = [-l for l in member_company_means] + [-l for l in nonmember_company_mean_list]
    auroc_company = roc_auc_score(y_true_company, y_scores_company) if len(set(y_true_company)) > 1 else None

    result = {
        "nonmember_mean_loss": nonmember_mean,
        "auroc": float(auroc_company) if auroc_company is not None else None,
        "auroc_doc": float(auroc_doc) if auroc_doc is not None else None,
        "per_company": per_company,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    write_json(args.output, result)
    print(json.dumps({"auroc": result["auroc"], "num_companies": len(per_company)}, indent=2))


if __name__ == "__main__":
    main()
