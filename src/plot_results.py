import argparse
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns


def load_ratios(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    ratios = []
    for cik, v in data.get("per_company", {}).items():
        r = v.get("loss_ratio")
        if r is not None:
            ratios.append(r)
    return ratios


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--c1", required=True)
    p.add_argument("--c2", required=True)
    p.add_argument("--c3", required=True)
    p.add_argument("--c4", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    data = {
        "C1 student": load_ratios(args.c1),
        "C2 student": load_ratios(args.c2),
        "C3 student": load_ratios(args.c3),
        "C4 teacher": load_ratios(args.c4),
    }

    labels = []
    values = []
    for k, v in data.items():
        labels.extend([k] * len(v))
        values.extend(v)

    sns.set(style="whitegrid")
    plt.figure(figsize=(9, 4.5))
    sns.violinplot(x=labels, y=values, cut=0)
    plt.ylabel("Loss ratio (member / non-member)")
    plt.xlabel("")
    plt.tight_layout()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    plt.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()
