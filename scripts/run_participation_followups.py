#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
VENV_PY = REPO / ".venv" / "bin" / "python"
if VENV_PY.exists() and Path(sys.executable) != VENV_PY:
    os.execv(str(VENV_PY), [str(VENV_PY), str(Path(__file__).resolve()), *sys.argv[1:]])

import numpy as np


EXP = ROOT / "exp"
PYTHIA_ROOT = EXP / "outputs" / "pythia-1.4b"
NEO_ROOT = EXP / "outputs" / "myriad_c23_repair_20260421" / "outputs" / "gpt-neo-1.3b-local"
DEFAULT_OUT = EXP / "outputs" / "attack_followups"


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text())


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def auc(pos: List[float], neg: List[float]) -> float:
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


def bootstrap_auc(pos: List[float], neg: List[float], n_boot: int = 2000, seed: int = 13) -> List[float]:
    rng = random.Random(seed)
    vals = []
    for _ in range(n_boot):
        pos_sample = [rng.choice(pos) for _ in pos]
        neg_sample = [rng.choice(neg) for _ in neg]
        vals.append(auc(pos_sample, neg_sample))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def bootstrap_mean_diff(values: List[float], n_boot: int = 2000, seed: int = 13) -> List[float]:
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        sample = [rng.choice(values) for _ in values]
        means.append(float(np.mean(sample)))
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def exact_sign_test(diffs: List[float]) -> Dict:
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return {"pos": 0, "neg": 0, "n": 0, "p_value_two_sided": 1.0}
    tail = sum(math.comb(n, k) for k in range(0, min(pos, neg) + 1)) / (2 ** n)
    p = min(1.0, 2.0 * tail)
    return {"pos": pos, "neg": neg, "n": n, "p_value_two_sided": float(p)}


def average_precision(scores: List[float], labels: List[int]) -> float:
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    tp = 0
    precision_sum = 0.0
    positives = sum(labels)
    if positives == 0:
        return float("nan")
    for rank, idx in enumerate(order, start=1):
        if labels[idx] == 1:
            tp += 1
            precision_sum += tp / rank
    return float(precision_sum / positives)


def threshold_metrics(pos: List[float], neg: List[float], fpr_targets: Iterable[float] = (0.01, 0.05, 0.10)) -> Dict:
    scores = pos + neg
    labels = [1] * len(pos) + [0] * len(neg)
    unique_thresholds = sorted(set(scores), reverse=True)
    thresholds = [float("inf")] + unique_thresholds + [float("-inf")]
    rows = []
    best_bal = None
    best_bal_row = None
    best_for_fpr = {x: None for x in fpr_targets}

    for thr in thresholds:
        preds = [1 if s >= thr else 0 for s in scores]
        tp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 1)
        fp = sum(1 for p, y in zip(preds, labels) if p == 1 and y == 0)
        tn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 0)
        fn = sum(1 for p, y in zip(preds, labels) if p == 0 and y == 1)
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        bal = 0.5 * (tpr + tnr)
        row = {
            "threshold": thr,
            "tpr": float(tpr),
            "fpr": float(fpr),
            "balanced_accuracy": float(bal),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }
        rows.append(row)
        if best_bal is None or bal > best_bal:
            best_bal = bal
            best_bal_row = row
        for target in fpr_targets:
            if fpr <= target:
                current = best_for_fpr[target]
                if current is None or tpr > current["tpr"]:
                    best_for_fpr[target] = row

    sorted_pairs = sorted(zip(scores, labels), reverse=True)
    topk = {}
    for k in (1, 5, 10):
        if len(sorted_pairs) < k:
            continue
        top = sorted_pairs[:k]
        hits = sum(label for _, label in top)
        topk[f"precision_at_{k}"] = float(hits / k)
        topk[f"recall_at_{k}"] = float(hits / len(pos))

    return {
        "auroc": auc(pos, neg),
        "average_precision": average_precision(scores, labels),
        "best_balanced_accuracy": best_bal_row,
        "tpr_at_fpr": {str(k): best_for_fpr[k] for k in fpr_targets},
        "topk": topk,
    }


def load_attack_scores(path: Path) -> Tuple[Dict[str, float], Dict[str, float], Dict]:
    data = read_json(path)
    return (
        {str(k): float(v) for k, v in data["target_delta_by_company"].items()},
        {str(k): float(v) for k, v in data["retain_delta_by_company"].items()},
        data,
    )


def load_target_mean_losses(path: Path) -> Dict[str, float]:
    data = read_json(path)
    return {str(k): float(v["mean_loss"]) for k, v in data["per_company"].items()}


def load_retain_mean_losses(path: Path) -> Dict[str, float]:
    data = read_json(path)
    return {str(k): float(v["mean_loss"]) for k, v in data["per_company"].items()}


def build_attack_from_losses(c6_target: Dict[str, float], ref_target: Dict[str, float], c6_retain: Dict[str, float], ref_retain: Dict[str, float]) -> Dict:
    target_keys = sorted(set(c6_target) & set(ref_target))
    retain_keys = sorted(set(c6_retain) & set(ref_retain))
    target_delta = {k: float(c6_target[k] - ref_target[k]) for k in target_keys}
    retain_delta = {k: float(c6_retain[k] - ref_retain[k]) for k in retain_keys}
    pos = list(target_delta.values())
    neg = list(retain_delta.values())
    return {
        "target_delta_by_company": target_delta,
        "retain_delta_by_company": retain_delta,
        "num_targets": len(pos),
        "num_retained": len(neg),
        "auroc": auc(pos, neg),
        "bootstrap_ci_95": bootstrap_auc(pos, neg),
        "target_mean_delta": float(np.mean(pos)),
        "retain_mean_delta": float(np.mean(neg)),
    }


def canonical_attack_specs() -> List[Dict]:
    return [
        {"scale": "pythia-1.4B", "actor": "student", "reference": "c3", "path": PYTHIA_ROOT / "mia_c6_deletion_attack_target_vs_retain.json"},
        {"scale": "pythia-1.4B", "actor": "student", "reference": "c1", "path": PYTHIA_ROOT / "mia_c6_deletion_attack_target_vs_retain_c1ref.json"},
        {"scale": "pythia-1.4B", "actor": "teacher", "reference": "c3", "path": PYTHIA_ROOT / "mia_c6_teacher_deletion_attack_target_vs_retain_c3ref.json"},
        {"scale": "pythia-1.4B", "actor": "teacher", "reference": "c1", "path": PYTHIA_ROOT / "mia_c6_teacher_deletion_attack_target_vs_retain_c1ref.json"},
        {"scale": "neo-1.3B", "actor": "student", "reference": "c3", "path": NEO_ROOT / "mia_c6_deletion_attack_target_vs_retain.json"},
        {"scale": "neo-1.3B", "actor": "student", "reference": "c1", "path": NEO_ROOT / "mia_c6_deletion_attack_target_vs_retain_c1ref.json"},
        {"scale": "neo-1.3B", "actor": "teacher", "reference": "c3", "path": NEO_ROOT / "mia_c6_teacher_deletion_attack_target_vs_retain_c3ref.json"},
        {"scale": "neo-1.3B", "actor": "teacher", "reference": "c1", "path": NEO_ROOT / "mia_c6_teacher_deletion_attack_target_vs_retain_c1ref.json"},
    ]


def compute_thresholded_results() -> List[Dict]:
    rows = []
    for spec in canonical_attack_specs():
        path = spec["path"]
        if not path.exists():
            continue
        target_scores, retain_scores, raw = load_attack_scores(path)
        metrics = threshold_metrics(list(target_scores.values()), list(retain_scores.values()))
        rows.append({
            "scale": spec["scale"],
            "actor": spec["actor"],
            "reference": spec["reference"],
            "auroc": raw.get("auroc"),
            "bootstrap_ci_95": raw.get("bootstrap_ci_95"),
            "threshold_metrics": metrics,
            "file": str(path),
        })
    return rows


def paired_auroc_gap(student_target: Dict[str, float], student_retain: Dict[str, float], teacher_target: Dict[str, float], teacher_retain: Dict[str, float], n_boot: int = 2000, n_perm: int = 20000, seed: int = 13) -> Dict:
    pos_keys = sorted(set(student_target) & set(teacher_target))
    neg_keys = sorted(set(student_retain) & set(teacher_retain))

    s_pos = [student_target[k] for k in pos_keys]
    s_neg = [student_retain[k] for k in neg_keys]
    t_pos = [teacher_target[k] for k in pos_keys]
    t_neg = [teacher_retain[k] for k in neg_keys]

    student_auc = auc(s_pos, s_neg)
    teacher_auc = auc(t_pos, t_neg)
    obs_gap = student_auc - teacher_auc

    rng = random.Random(seed)
    boot = []
    for _ in range(n_boot):
        pos_sample = [rng.choice(pos_keys) for _ in pos_keys]
        neg_sample = [rng.choice(neg_keys) for _ in neg_keys]
        s_auc = auc([student_target[k] for k in pos_sample], [student_retain[k] for k in neg_sample])
        t_auc = auc([teacher_target[k] for k in pos_sample], [teacher_retain[k] for k in neg_sample])
        boot.append(s_auc - t_auc)

    count = 0
    for _ in range(n_perm):
        s_pos_perm = []
        t_pos_perm = []
        for k in pos_keys:
            a, b = student_target[k], teacher_target[k]
            if rng.random() < 0.5:
                s_pos_perm.append(a)
                t_pos_perm.append(b)
            else:
                s_pos_perm.append(b)
                t_pos_perm.append(a)
        s_neg_perm = []
        t_neg_perm = []
        for k in neg_keys:
            a, b = student_retain[k], teacher_retain[k]
            if rng.random() < 0.5:
                s_neg_perm.append(a)
                t_neg_perm.append(b)
            else:
                s_neg_perm.append(b)
                t_neg_perm.append(a)
        perm_gap = auc(s_pos_perm, s_neg_perm) - auc(t_pos_perm, t_neg_perm)
        if abs(perm_gap) >= abs(obs_gap):
            count += 1

    # Company-level score difference summaries are included to show systematic shift.
    target_score_gap = [student_target[k] - teacher_target[k] for k in pos_keys]
    retain_score_gap = [student_retain[k] - teacher_retain[k] for k in neg_keys]

    return {
        "n_positive": len(pos_keys),
        "n_negative": len(neg_keys),
        "student_auroc": student_auc,
        "teacher_auroc": teacher_auc,
        "student_minus_teacher_auroc": obs_gap,
        "bootstrap_ci_95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
        "permutation_test": {
            "obs_gap": obs_gap,
            "p_value_two_sided": float((count + 1) / (n_perm + 1)),
            "n": len(pos_keys) + len(neg_keys),
        },
        "target_score_gap_mean": float(np.mean(target_score_gap)),
        "retain_score_gap_mean": float(np.mean(retain_score_gap)),
    }


def compute_student_teacher_gap_results() -> Dict:
    canonical = []
    for scale, base in [("pythia-1.4B", PYTHIA_ROOT), ("neo-1.3B", NEO_ROOT)]:
        for ref in ["c1", "c3"]:
            student_path = base / ("mia_c6_deletion_attack_target_vs_retain_c1ref.json" if ref == "c1" else "mia_c6_deletion_attack_target_vs_retain.json")
            teacher_path = base / f"mia_c6_teacher_deletion_attack_target_vs_retain_{'c1ref' if ref == 'c1' else 'c3ref'}.json"
            if not student_path.exists() or not teacher_path.exists():
                continue
            s_target, s_retain, _ = load_attack_scores(student_path)
            t_target, t_retain, _ = load_attack_scores(teacher_path)
            canonical.append({
                "scale": scale,
                "reference": ref,
                "student_file": str(student_path),
                "teacher_file": str(teacher_path),
                **paired_auroc_gap(s_target, s_retain, t_target, t_retain),
            })

    seeded = []
    for ref in ["c1", "c3"]:
        diffs = []
        per_seed = []
        for seed in [13, 17, 19]:
            base = PYTHIA_ROOT / "seed_reps" / f"seed_{seed}"
            student_path = base / ("mia_c6_deletion_attack_target_vs_retain_c1ref.json" if ref == "c1" else "mia_c6_deletion_attack_target_vs_retain.json")
            teacher_path = base / (
                "mia_c6_teacher_deletion_attack_target_vs_retain_c1ref.json"
                if ref == "c1"
                else "mia_c6_teacher_deletion_attack_target_vs_retain.json"
            )
            if not student_path.exists() or not teacher_path.exists():
                continue
            s_target, s_retain, _ = load_attack_scores(student_path)
            t_target, t_retain, _ = load_attack_scores(teacher_path)
            gap = paired_auroc_gap(s_target, s_retain, t_target, t_retain, n_boot=1000, n_perm=5000)
            per_seed.append({"seed": seed, **gap})
            diffs.append(gap["student_minus_teacher_auroc"])
        if diffs:
            seeded.append({
                "scale": "pythia-1.4B",
                "reference": ref,
                "per_seed": per_seed,
                "mean_student_minus_teacher_auroc": float(np.mean(diffs)),
                "bootstrap_ci_95": bootstrap_mean_diff(diffs),
                "sign_test": exact_sign_test(diffs),
            })

    return {"canonical": canonical, "seeded_pythia": seeded}


def compute_reference_mismatch_results() -> Dict:
    rows = []
    for c6_seed in [13, 17, 19]:
        base = PYTHIA_ROOT / "seed_reps" / f"seed_{c6_seed}"
        c6_target = load_target_mean_losses(base / "mia" / "c6_student.json")
        c6_retain = load_retain_mean_losses(base / "mia_retain" / "c6_student_retain.json")

        # Requested baseline references.
        for ref_name, ref_seed in [("c3_same_seed", c6_seed), ("c1_same_seed", c6_seed)]:
            ref_base = PYTHIA_ROOT / "seed_reps" / f"seed_{ref_seed}"
            ref_cond = "c3" if ref_name.startswith("c3") else "c1"
            ref_target = load_target_mean_losses(ref_base / "mia" / f"{ref_cond}_student.json")
            ref_retain = load_retain_mean_losses(ref_base / "mia_retain" / f"{ref_cond}_student_retain.json")
            attack = build_attack_from_losses(c6_target, ref_target, c6_retain, ref_retain)
            rows.append({
                "scale": "pythia-1.4B",
                "c6_seed": c6_seed,
                "reference_kind": ref_name,
                "reference_seed": ref_seed,
                "reference_condition": ref_cond,
                "attack": attack,
                "threshold_metrics": threshold_metrics(
                    list(attack["target_delta_by_company"].values()),
                    list(attack["retain_delta_by_company"].values()),
                ),
            })

        for ref_seed in [13, 17, 19]:
            if ref_seed == c6_seed:
                continue
            ref_base = PYTHIA_ROOT / "seed_reps" / f"seed_{ref_seed}"
            ref_target = load_target_mean_losses(ref_base / "mia" / "c1_student.json")
            ref_retain = load_retain_mean_losses(ref_base / "mia_retain" / "c1_student_retain.json")
            attack = build_attack_from_losses(c6_target, ref_target, c6_retain, ref_retain)
            rows.append({
                "scale": "pythia-1.4B",
                "c6_seed": c6_seed,
                "reference_kind": "c1_cross_seed",
                "reference_seed": ref_seed,
                "reference_condition": "c1",
                "attack": attack,
                "threshold_metrics": threshold_metrics(
                    list(attack["target_delta_by_company"].values()),
                    list(attack["retain_delta_by_company"].values()),
                ),
            })

    grouped = {}
    for row in rows:
        key = row["reference_kind"]
        grouped.setdefault(key, []).append(row["attack"]["auroc"])

    unavailable = [{
        "reference_kind": "same_teacher_different_distill_seed",
        "status": "not_computed",
        "reason": "No alternate final student references from the same teacher with a different distillation seed are present in the current artifacts.",
    }]

    return {
        "rows": rows,
        "summary": {
            key: {
                "mean_auroc": float(np.mean(vals)),
                "sd_auroc": float(np.std(vals)),
                "n": len(vals),
            }
            for key, vals in grouped.items()
        },
        "unavailable": unavailable,
    }


def placebo_status(placebo_json: Path = None) -> Dict:
    if placebo_json is None:
        return {
            "status": "not_run",
            "reason": "Placebo/random-forget NPO is a new training branch. This script does not fabricate it from existing artifacts.",
            "required_inputs": [
                "placebo teacher checkpoint",
                "placebo student checkpoint or placebo attack json",
                "attack eval against original target/retain split",
            ],
        }
    data = read_json(placebo_json)
    return {
        "status": "loaded",
        "file": str(placebo_json),
        "auroc": data.get("auroc"),
        "bootstrap_ci_95": data.get("bootstrap_ci_95"),
    }


def build_summary_md(results: Dict) -> str:
    lines = ["# Participation Follow-ups", ""]

    lines.append("## Thresholded Performance")
    for row in results["threshold_metrics"]:
        metric = row["threshold_metrics"]
        ba = metric["best_balanced_accuracy"]
        lines.append(
            f"- {row['scale']} {row['actor']} ref={row['reference']}: AUROC {row['auroc']:.4f}; "
            f"TPR@1%FPR {metric['tpr_at_fpr']['0.01']['tpr'] if metric['tpr_at_fpr']['0.01'] else float('nan'):.4f}; "
            f"TPR@5%FPR {metric['tpr_at_fpr']['0.05']['tpr'] if metric['tpr_at_fpr']['0.05'] else float('nan'):.4f}; "
            f"TPR@10%FPR {metric['tpr_at_fpr']['0.1']['tpr'] if metric['tpr_at_fpr']['0.1'] else float('nan'):.4f}; "
            f"best balanced acc {ba['balanced_accuracy']:.4f}; "
            f"P@5 {metric['topk'].get('precision_at_5', float('nan')):.4f}."
        )
    lines.append("")

    lines.append("## Student-Teacher Gap")
    for row in results["student_teacher_gap"]["canonical"]:
        ci = row["bootstrap_ci_95"]
        lines.append(
            f"- {row['scale']} ref={row['reference']}: student-teacher AUROC gap {row['student_minus_teacher_auroc']:.4f} "
            f"[{ci[0]:.4f}, {ci[1]:.4f}], p={row['permutation_test']['p_value_two_sided']:.6f}."
        )
    for row in results["student_teacher_gap"]["seeded_pythia"]:
        ci = row["bootstrap_ci_95"]
        lines.append(
            f"- seeded Pythia ref={row['reference']}: mean gap {row['mean_student_minus_teacher_auroc']:.4f} "
            f"[{ci[0]:.4f}, {ci[1]:.4f}], sign test p={row['sign_test']['p_value_two_sided']:.6f}."
        )
    lines.append("")

    lines.append("## Reference Mismatch")
    for key, summary in results["reference_mismatch"]["summary"].items():
        lines.append(
            f"- {key}: mean AUROC {summary['mean_auroc']:.4f}, sd {summary['sd_auroc']:.4f}, n={summary['n']}."
        )
    for row in results["reference_mismatch"]["unavailable"]:
        lines.append(f"- {row['reference_kind']}: {row['reason']}")
    lines.append("")

    lines.append("## Placebo")
    placebo = results["placebo_control"]
    if placebo["status"] == "loaded":
        lines.append(f"- loaded {placebo['file']}: AUROC {placebo['auroc']}.")
    else:
        lines.append(f"- not run: {placebo['reason']}")

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--placebo-json", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "threshold_metrics": compute_thresholded_results(),
        "student_teacher_gap": compute_student_teacher_gap_results(),
        "reference_mismatch": compute_reference_mismatch_results(),
        "placebo_control": placebo_status(args.placebo_json),
    }

    write_json(out_dir / "threshold_metrics.json", {"rows": results["threshold_metrics"]})
    write_json(out_dir / "student_teacher_gap.json", results["student_teacher_gap"])
    write_json(out_dir / "reference_mismatch.json", results["reference_mismatch"])
    write_json(out_dir / "placebo_control.json", results["placebo_control"])
    (out_dir / "summary.md").write_text(build_summary_md(results))
    print(f"Wrote follow-up diagnostics under {out_dir}")


if __name__ == "__main__":
    main()
