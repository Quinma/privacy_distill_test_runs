import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import datasets
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer

from data_prep import _get_fields, _form_matches, _load_cik_map, _load_streaming_dataset
from utils import clean_edgar_text, write_json


def _load_stats(stats_path: str) -> Dict[str, Dict]:
    rows = {}
    with open(stats_path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            cik = str(r.get("cik", ""))
            if cik:
                rows[cik] = r
    return rows


def _parse_date(d: str):
    if not d:
        return None
    try:
        return datetime.fromisoformat(d)
    except Exception:
        return None


def _span_days(r: Dict) -> float:
    d0 = _parse_date(r.get("min_date"))
    d1 = _parse_date(r.get("max_date"))
    if not d0 or not d1:
        return 0.0
    return max(0.0, (d1 - d0).days)


def _match_candidates(
    targets: List[str],
    stats: Dict[str, Dict],
    candidates: List[str],
    seed: int,
    weight_tokens: float = 1.0,
    weight_filings: float = 0.1,
    weight_span: float = 0.02,
) -> List[Tuple[str, str]]:
    rng = np.random.RandomState(seed)
    cand_set = {c for c in candidates if c in stats}
    # shuffle to avoid deterministic tie bias
    cand_list = list(cand_set)
    rng.shuffle(cand_list)

    matched = []
    used = set()
    for t in targets:
        t_stat = stats.get(t)
        if not t_stat:
            continue
        t_tok = float(t_stat.get("token_count", 0))
        t_fil = float(t_stat.get("num_filings", 0))
        t_span = _span_days(t_stat)
        best = None
        best_score = None
        for c in cand_list:
            if c in used:
                continue
            c_stat = stats.get(c)
            if not c_stat:
                continue
            c_tok = float(c_stat.get("token_count", 0))
            c_fil = float(c_stat.get("num_filings", 0))
            c_span = _span_days(c_stat)
            # weighted distance in log/token space to reduce scale effects
            tok_score = abs(np.log1p(t_tok) - np.log1p(c_tok))
            fil_score = abs(np.log1p(t_fil) - np.log1p(c_fil))
            span_score = abs(t_span - c_span) / 3650.0  # normalize to ~decades
            score = weight_tokens * tok_score + weight_filings * fil_score + weight_span * span_score
            if best_score is None or score < best_score:
                best_score = score
                best = c
        if best is not None:
            used.add(best)
            matched.append((t, best))
    return matched


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="bradfordlevy/BeanCounter")
    p.add_argument("--config", default="clean")
    p.add_argument("--revision", default=None)
    p.add_argument("--split", default="train")
    p.add_argument("--data-files", default=None)
    p.add_argument("--cik-map", default=None)
    p.add_argument("--tokenizer", default="EleutherAI/pythia-1.4b")
    p.add_argument("--form-types", default="10-K")
    p.add_argument("--text-field", default="text")
    p.add_argument("--cik-field", default="cik")
    p.add_argument("--form-field", default="type_filing")
    p.add_argument("--date-field", default="date")
    p.add_argument("--stats-path", default="data/bean_counter_stats.jsonl")
    p.add_argument("--splits", required=True)
    p.add_argument("--eval-nonmember", default=None, help="Existing eval_nonmember dataset; used to restrict candidates")
    p.add_argument("--nonmember-tokens", type=int, default=5_000_000)
    p.add_argument("--max-docs", type=int, default=0)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--output", required=True)
    p.add_argument("--save-matched-json", default=None)
    return p


def main():
    args = build_parser().parse_args()

    with open(args.splits, "r", encoding="utf-8") as f:
        splits = json.load(f)
    targets = [str(x) for x in splits.get("targets", [])]
    retain = [str(x) for x in splits.get("retain", [])]
    selected = set(targets + retain)

    stats = _load_stats(args.stats_path)

    candidates = []
    if args.eval_nonmember and os.path.exists(args.eval_nonmember):
        try:
            ds = datasets.load_from_disk(args.eval_nonmember)
            ciks = sorted({str(ex.get("cik", "")) for ex in ds if ex.get("cik", "")})
            candidates = [c for c in ciks if c and c not in selected]
        except Exception:
            candidates = []

    if not candidates:
        # fallback: all non-selected companies in stats
        candidates = [c for c in stats.keys() if c not in selected]

    matched_pairs = _match_candidates(targets, stats, candidates, seed=args.seed)
    matched_ciks = [c for _, c in matched_pairs]

    if args.save_matched_json:
        write_json(args.save_matched_json, {
            "targets": targets,
            "matched_pairs": matched_pairs,
            "matched_ciks": matched_ciks,
        })

    # Build matched nonmember eval dataset
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    ds = _load_streaming_dataset(args)
    cik_map = _load_cik_map(args.cik_map)
    include_forms = set([f.strip().upper() for f in args.form_types.split(",")])

    matched_set = set(matched_ciks)
    docs = []
    token_budget = args.nonmember_tokens
    seen_ciks = set()

    for ex in ds:
        text, cik, form, date = _get_fields(ex, {
            "text": args.text_field,
            "cik": args.cik_field,
            "form": args.form_field,
            "date": args.date_field,
        }, cik_map)
        accession = ex.get("accession")
        if not cik or cik not in matched_set:
            continue
        if include_forms and not _form_matches(form, include_forms):
            continue
        text = clean_edgar_text(text)
        if not text:
            continue
        tok = len(tokenizer.encode(text))
        docs.append({
            "cik": cik,
            "text": text,
            "form": form,
            "date": date,
            "accession": accession,
        })
        seen_ciks.add(cik)
        token_budget -= tok
        if args.max_docs and len(docs) >= args.max_docs:
            break
        # stop only once token budget reached AND we have at least 1 doc per cik
        if token_budget <= 0 and len(seen_ciks) == len(matched_set):
            break

    if not docs:
        raise SystemExit("No matched nonmember docs collected.")

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    ds_out = Dataset.from_list(docs)
    if os.path.exists(out_dir):
        import shutil
        shutil.rmtree(out_dir)
    ds_out.save_to_disk(out_dir)

    summary = {
        "matched_ciks": len(matched_ciks),
        "matched_docs": len(docs),
        "matched_seen_ciks": len(seen_ciks),
        "nonmember_tokens_target": args.nonmember_tokens,
        "nonmember_tokens_remaining": token_budget,
    }
    write_json(os.path.join(out_dir, "matched_summary.json"), summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
