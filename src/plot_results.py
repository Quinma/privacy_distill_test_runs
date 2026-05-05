import argparse
import json
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def _load_scores(path: str):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    per = obj.get("per_company", {})
    scores = []
    for _, v in per.items():
        r = v.get("loss_ratio")
        if r is not None:
            scores.append(r)
    return np.array(scores, dtype=float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c1", required=True)
    p.add_argument("--c2", required=True)
    p.add_argument("--c3", required=True)
    p.add_argument("--c4", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--c5", default=None)
    args = p.parse_args()

    labels = ["C4 Teacher", "C1 Student", "C2 Student", "C3 Student"]
    data = [
        _load_scores(args.c4),
        _load_scores(args.c1),
        _load_scores(args.c2),
        _load_scores(args.c3),
    ]

    if args.c5:
        labels.append("C5 Student (Approx Unlearn)")
        data.append(_load_scores(args.c5))

    fig, ax = plt.subplots(figsize=(10, 5))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)

    for pc in parts["bodies"]:
        pc.set_facecolor("#4C78A8")
        pc.set_edgecolor("black")
        pc.set_alpha(0.6)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Loss ratio (member / non-member)")
    ax.set_title("Per-company MIA Loss Ratios")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
