#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np


def load_company_losses(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text())
    per_company = data.get("per_company", {})
    out = {}
    for cik, row in per_company.items():
        if isinstance(row, dict) and "mean_loss" in row:
            out[str(cik)] = float(row["mean_loss"])
    return out


def auc(pos: List[float], neg: List[float]) -> float:
    scores = [(v, 1) for v in pos] + [(v, 0) for v in neg]
    scores.sort(key=lambda x: x[0])
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


def bootstrap_auc(pos: List[float], neg: List[float], seed: int = 13, n_boot: int = 2000) -> List[float]:
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        pos_sample = [rng.choice(pos) for _ in pos]
        neg_sample = [rng.choice(neg) for _ in neg]
        vals.append(auc(pos_sample, neg_sample))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def build_result(c6_target, ref_target, c6_retain, ref_retain, reference: str, seed: int, control: str) -> dict:
    target_keys = sorted(set(c6_target) & set(ref_target))
    retain_keys = sorted(set(c6_retain) & set(ref_retain))
    target_delta = {k: c6_target[k] - ref_target[k] for k in target_keys}
    retain_delta = {k: c6_retain[k] - ref_retain[k] for k in retain_keys}
    target_vals = list(target_delta.values())
    retain_vals = list(retain_delta.values())
    retain_mean = float(np.mean(retain_vals))
    return {
        "seed": seed,
        "control": control,
        "attack": "deletion_target_inference",
        "reference": reference,
        "positive_class": "deleted_targets",
        "negative_class": "retained_companies",
        "num_targets": len(target_vals),
        "num_retained": len(retain_vals),
        "auroc": auc(target_vals, retain_vals),
        "bootstrap_ci_95": bootstrap_auc(target_vals, retain_vals, seed=seed),
        "target_mean_delta": float(np.mean(target_vals)),
        "retain_mean_delta": retain_mean,
        "target_median_delta": float(np.median(target_vals)),
        "retain_median_delta": float(np.median(retain_vals)),
        "num_target_gt_retain_mean": int(sum(1 for x in target_vals if x > retain_mean)),
        "target_delta_by_company": target_delta,
        "retain_delta_by_company": retain_delta,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c6-target", required=True)
    parser.add_argument("--ref-target", required=True)
    parser.add_argument("--c6-retain", required=True)
    parser.add_argument("--ref-retain", required=True)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--control", default="canonical_c6_npo")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    result = build_result(
        load_company_losses(Path(args.c6_target)),
        load_company_losses(Path(args.ref_target)),
        load_company_losses(Path(args.c6_retain)),
        load_company_losses(Path(args.ref_retain)),
        args.reference,
        args.seed,
        args.control,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
