import argparse
import json
import os
import random
from typing import Dict, List, Tuple

import numpy as np


def read_json(path: str) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def bootstrap_ci(values: List[float], n: int = 2000, seed: int = 13) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    vals = values[:]
    means = []
    for _ in range(n):
        sample = [rng.choice(vals) for _ in range(len(vals))]
        means.append(float(np.mean(sample)))
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def bootstrap_auroc(member_losses: List[float], nonmember_losses: List[float], n: int = 2000, seed: int = 13) -> Tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    rng = random.Random(seed)
    m = member_losses[:]
    nmem = nonmember_losses[:]
    if not m or not nmem:
        return (float("nan"), float("nan"))
    aurocs = []
    for _ in range(n):
        m_s = [rng.choice(m) for _ in range(len(m))]
        n_s = [rng.choice(nmem) for _ in range(len(nmem))]
        y_true = [1] * len(m_s) + [0] * len(n_s)
        y_scores = [-x for x in m_s] + [-x for x in n_s]
        aurocs.append(float(roc_auc_score(y_true, y_scores)))
    return (float(np.percentile(aurocs, 2.5)), float(np.percentile(aurocs, 97.5)))


def paired_permutation_test(x: List[float], y: List[float], n: int = 20000, seed: int = 13) -> Dict:
    rng = random.Random(seed)
    diffs = [a - b for a, b in zip(x, y)]
    obs = float(np.mean(diffs))
    count = 0
    for _ in range(n):
        flipped = [d if rng.random() < 0.5 else -d for d in diffs]
        if abs(np.mean(flipped)) >= abs(obs):
            count += 1
    p = (count + 1) / (n + 1)
    return {"obs_mean_diff": obs, "p_value_two_sided": p, "n": len(diffs)}


def sign_test(x: List[float], y: List[float]) -> Dict:
    from scipy.stats import binomtest
    diffs = [a - b for a, b in zip(x, y)]
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return {"pos": 0, "neg": 0, "n": 0, "p_value_two_sided": 1.0}
    p = binomtest(pos, n, 0.5, alternative="two-sided").pvalue
    return {"pos": pos, "neg": neg, "n": n, "p_value_two_sided": float(p)}


def bootstrap_ci_diff(x: List[float], y: List[float], n: int = 2000, seed: int = 13) -> Tuple[float, float]:
    rng = random.Random(seed)
    diffs = [a - b for a, b in zip(x, y)]
    if not diffs:
        return (float("nan"), float("nan"))
    means = []
    for _ in range(n):
        sample = [rng.choice(diffs) for _ in range(len(diffs))]
        means.append(float(np.mean(sample)))
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def load_condition(path: str) -> Dict:
    data = read_json(path)
    per_company = data["per_company"]
    loss_ratio = {
        k: v.get("loss_ratio") for k, v in per_company.items() if v.get("loss_ratio") is not None
    }
    mean_loss = {
        k: v.get("mean_loss") for k, v in per_company.items() if v.get("mean_loss") is not None
    }
    nonmember_means = list(data.get("nonmember_company_means", {}).values())
    return {
        "auroc": data.get("auroc"),
        "auroc_doc": data.get("auroc_doc"),
        "loss_ratio": loss_ratio,
        "mean_loss": mean_loss,
        "nonmember_company_means": nonmember_means,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c1", required=True)
    p.add_argument("--c2", required=True)
    p.add_argument("--c3", required=True)
    p.add_argument("--c4", required=True)
    p.add_argument("--c5", default=None)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--boot", type=int, default=2000)
    p.add_argument("--perm", type=int, default=20000)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    conds = {
        "C1": load_condition(args.c1),
        "C2": load_condition(args.c2),
        "C3": load_condition(args.c3),
        "C4": load_condition(args.c4),
    }
    if args.c5:
        conds["C5"] = load_condition(args.c5)

    # Bootstrap CIs per condition (AUROC + mean loss_ratio)
    stats = {}
    for name, d in conds.items():
        lr_vals = list(d["loss_ratio"].values())
        m_losses = list(d["mean_loss"].values())
        nm_losses = list(d["nonmember_company_means"])
        ci_lr = bootstrap_ci(lr_vals, n=args.boot)
        ci_auc = bootstrap_auroc(m_losses, nm_losses, n=args.boot)
        stats[name] = {
            "auroc": d["auroc"],
            "auroc_doc": d["auroc_doc"],
            "mean_loss_ratio": float(np.mean(lr_vals)) if lr_vals else None,
            "loss_ratio_ci95": [ci_lr[0], ci_lr[1]],
            "auroc_ci95": [ci_auc[0], ci_auc[1]],
            "n_companies": len(lr_vals),
            "n_nonmember_companies": len(nm_losses),
        }

    def paired_stats(label_a: str, label_b: str) -> Dict:
        # Paired stats on per-company mean_loss (raw loss), not loss_ratio.
        common = sorted(set(conds[label_a]["mean_loss"]) & set(conds[label_b]["mean_loss"]))
        a_vals = [conds[label_a]["mean_loss"][k] for k in common]
        b_vals = [conds[label_b]["mean_loss"][k] for k in common]
        return {
            "n": len(common),
            "sign_test": sign_test(a_vals, b_vals),
            "perm_test": paired_permutation_test(a_vals, b_vals, n=args.perm),
            "bootstrap_ci_diff": list(bootstrap_ci_diff(a_vals, b_vals, n=args.boot)),
            "mean_diff": float(np.mean([a - b for a, b in zip(a_vals, b_vals)])) if common else None,
        }

    paired = {
        "C2_vs_C3": paired_stats("C2", "C3"),
        "C1_vs_C3": paired_stats("C1", "C3"),
    }
    if "C5" in conds:
        paired["C5_vs_C3"] = paired_stats("C5", "C3")

    out = {
        "bootstrap": stats,
        "paired": paired,
    }

    with open(os.path.join(args.out_dir, "stats_bootstrap.json"), "w") as f:
        json.dump(out, f, indent=2)

    # Summary table (CSV + Markdown)
    rows = []
    for name, s in stats.items():
        rows.append({
            "condition": name,
            "auroc": s["auroc"],
            "auroc_ci95_low": s["auroc_ci95"][0],
            "auroc_ci95_high": s["auroc_ci95"][1],
            "auroc_doc": s["auroc_doc"],
            "mean_loss_ratio": s["mean_loss_ratio"],
            "loss_ratio_ci95_low": s["loss_ratio_ci95"][0],
            "loss_ratio_ci95_high": s["loss_ratio_ci95"][1],
            "n_companies": s["n_companies"],
            "n_nonmember_companies": s["n_nonmember_companies"],
        })

    import csv
    csv_path = os.path.join(args.out_dir, "summary_table.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_path = os.path.join(args.out_dir, "summary_table.md")
    with open(md_path, "w") as f:
        f.write("| " + " | ".join(rows[0].keys()) + " |\\n")
        f.write("| " + " | ".join(["---"] * len(rows[0])) + " |\\n")
        for r in rows:
            f.write("| " + " | ".join(str(r[k]) for k in rows[0].keys()) + " |\\n")


if __name__ == "__main__":
    main()
