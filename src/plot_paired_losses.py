import argparse
import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def load_losses(path: str, eval_set: str) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Returns: data[scale][condition][cik] = mean_loss
    """
    data: Dict[str, Dict[str, Dict[str, float]]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("eval_set") != eval_set:
                continue
            scale = row.get("scale")
            cond = row.get("condition")
            cik = row.get("cik")
            mean_loss = row.get("mean_loss")
            if not scale or not cond or not cik or mean_loss is None or mean_loss == "":
                continue
            try:
                mean_loss = float(mean_loss)
            except Exception:
                continue
            data.setdefault(scale, {}).setdefault(cond, {})[cik] = mean_loss
    return data


def paired_points(a: Dict[str, float], b: Dict[str, float]) -> Tuple[List[float], List[float], List[str]]:
    common = sorted(set(a) & set(b))
    x = [a[c] for c in common]
    y = [b[c] for c in common]
    return x, y, common


def pad_range(vals: List[float], frac: float = 0.05) -> Tuple[float, float]:
    vmin = min(vals)
    vmax = max(vals)
    if vmax == vmin:
        pad = 0.1 * vmax if vmax != 0 else 0.1
    else:
        pad = (vmax - vmin) * frac
    return vmin - pad, vmax + pad


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--loss-csv", default="outputs/per_company_losses.csv")
    p.add_argument("--output", default="outputs/mia/figure2_paired_losses.png")
    p.add_argument("--eval-set", default="canonical", choices=["canonical", "matched"])
    p.add_argument("--include-2p8b", action="store_true")
    args = p.parse_args()

    data = load_losses(args.loss_csv, args.eval_set)

    # Always include 1.4B
    scales = ["1.4B"]
    if args.include_2p8b:
        scales.append("2.8B")

    # Colors and styles
    color_c1 = "#1a3a5c"  # navy
    color_c5 = "#c44e52"  # red

    # Matplotlib styling
    plt.rcParams.update({
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 4.0), constrained_layout=True)

    # LEFT: C1 vs C3
    ax = axes[0]
    x_all, y_all = [], []
    for scale in scales:
        if scale not in data:
            continue
        c1 = data[scale].get("C1", {})
        c3 = data[scale].get("C3", {})
        x, y, _ = paired_points(c3, c1)  # x=C3, y=C1
        if not x:
            continue
        if scale == "2.8B":
            ax.scatter(x, y, s=25, marker="o", facecolors="none", edgecolors=color_c1, alpha=0.7, linewidths=0.8)
        else:
            ax.scatter(x, y, s=25, marker="o", color=color_c1, alpha=0.7)
        x_all.extend(x)
        y_all.extend(y)

    if x_all and y_all:
        rng_min, rng_max = pad_range(x_all + y_all)
        ax.set_xlim(rng_min, rng_max)
        ax.set_ylim(rng_min, rng_max)
        ax.plot([rng_min, rng_max], [rng_min, rng_max], color="black", linewidth=0.6, alpha=0.4)

    ax.set_xlabel("C3 student mean loss (nats).")
    ax.set_ylabel("C1 student mean loss (nats).")
    ax.set_title("C1 vs C3 (memorisation)", fontweight="bold")
    ax.text(0.98, 0.02, "50/50 below diagonal", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, style="italic", color="#666666")

    # RIGHT: C5 vs C3
    ax = axes[1]
    x_all, y_all = [], []
    for scale in scales:
        if scale not in data:
            continue
        c5 = data[scale].get("C5", {})
        c3 = data[scale].get("C3", {})
        x, y, _ = paired_points(c3, c5)  # x=C3, y=C5
        if not x:
            continue
        if scale == "2.8B":
            ax.scatter(x, y, s=25, marker="v", facecolors="none", edgecolors=color_c5, alpha=0.7, linewidths=0.8)
        else:
            ax.scatter(x, y, s=25, marker="v", color=color_c5, alpha=0.7)
        x_all.extend(x)
        y_all.extend(y)

    if x_all and y_all:
        x_min, x_max = pad_range(x_all)
        y_min, y_max = pad_range(y_all)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        diag_min = min(x_min, y_min)
        diag_max = max(x_max, y_max)
        ax.plot([diag_min, diag_max], [diag_min, diag_max], color="black", linewidth=0.6, alpha=0.4)

    ax.set_xlabel("C3 student mean loss (nats).")
    ax.set_ylabel("C5 student mean loss (nats).")
    ax.set_title("C5 vs C3 (anti-memorisation)", fontweight="bold")
    ax.text(0.02, 0.98, "50/50 above diagonal", transform=ax.transAxes,
            ha="left", va="top", fontsize=7, style="italic", color="#666666")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=300)


if __name__ == "__main__":
    main()
