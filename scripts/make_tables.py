#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "repro_tables"


def load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text())


def add_row(rows, family, experiment, reference, path):
    data = load_json(path)
    if not data:
        return
    rows.append({
        "family": family,
        "experiment": experiment,
        "reference": reference,
        "auroc": data.get("auroc"),
        "ci_low": (data.get("bootstrap_ci_95") or [None, None])[0],
        "ci_high": (data.get("bootstrap_ci_95") or [None, None])[1],
        "target_mean_delta": data.get("target_mean_delta"),
        "retain_mean_delta": data.get("retain_mean_delta"),
        "file": str(path),
    })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["pythia-1.4b", "neo-1.3b", "all"], default="all")
    args = ap.parse_args()

    rows = []
    if args.family in {"pythia-1.4b", "all"}:
        base = ROOT / "outputs" / "pythia-1.4b"
        add_row(rows, "pythia-1.4b", "canonical_c6", "C1", base / "mia_c6_deletion_attack_target_vs_retain_c1ref.json")
        add_row(rows, "pythia-1.4b", "canonical_c6", "C3", base / "mia_c6_deletion_attack_target_vs_retain.json")
        placebo = ROOT / "outputs" / "pythia-1.4b-placebo-npo-s13"
        add_row(rows, "pythia-1.4b", "wrong_target_placebo", "C1", placebo / "mia_c6_placebo_deletion_attack_target_vs_retain_c1ref.json")
        add_row(rows, "pythia-1.4b", "wrong_target_placebo", "C3", placebo / "mia_c6_placebo_deletion_attack_target_vs_retain.json")
    if args.family in {"neo-1.3b", "all"}:
        base = ROOT / "outputs" / "gpt-neo-1.3b-local"
        add_row(rows, "neo-1.3b", "canonical_c6", "C1", base / "mia_c6_deletion_attack_target_vs_retain_c1ref.json")
        add_row(rows, "neo-1.3b", "canonical_c6", "C3", base / "mia_c6_deletion_attack_target_vs_retain.json")
        placebo = ROOT / "outputs" / "gpt-neo-1.3b-local-placebo-npo-s13"
        add_row(rows, "neo-1.3b", "wrong_target_placebo", "C1", placebo / "mia_c6_placebo_deletion_attack_target_vs_retain_c1ref.json")
        add_row(rows, "neo-1.3b", "wrong_target_placebo", "C3", placebo / "mia_c6_placebo_deletion_attack_target_vs_retain.json")

    OUT.mkdir(parents=True, exist_ok=True)
    stem = args.family.replace('.', '_')
    json_path = OUT / f"{stem}_summary.json"
    csv_path = OUT / f"{stem}_summary.csv"
    md_path = OUT / f"{stem}_summary.md"
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        with md_path.open("w") as f:
            f.write("| " + " | ".join(rows[0].keys()) + " |\n")
            f.write("| " + " | ".join(["---"] * len(rows[0])) + " |\n")
            for row in rows:
                f.write("| " + " | ".join(str(row[k]) for k in rows[0].keys()) + " |\n")
    else:
        csv_path.write_text("")
        md_path.write_text("No rows found.\n")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
