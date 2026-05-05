import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import datasets
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from utils import clean_edgar_text, dedupe_texts, set_seed, stable_hash, write_json, write_jsonl


def _extract_cik(example, field_map, cik_map: Dict[str, str]) -> str:
    cik = example.get(field_map["cik"])
    if cik:
        return str(cik)
    accession = example.get("accession")
    if accession and cik_map:
        return cik_map.get(str(accession), "")
    return ""


def _get_fields(example, field_map, cik_map: Dict[str, str]):
    text = example.get(field_map["text"], "")
    cik = _extract_cik(example, field_map, cik_map)
    form = example.get(field_map["form"], "")
    date = example.get(field_map["date"], "")
    return text, cik, form, date


def _form_matches(form_value: str, allowed: set) -> bool:
    if not form_value:
        return False
    form_value = str(form_value).upper()
    for f in allowed:
        if form_value.startswith(f):
            return True
    return False


def _load_streaming_dataset(args):
    data_files = None
    if args.data_files:
        if args.data_files.startswith("["):
            data_files = json.loads(args.data_files)
        elif "," in args.data_files:
            data_files = [s.strip() for s in args.data_files.split(",") if s.strip()]
        else:
            data_files = args.data_files
    return datasets.load_dataset(
        args.dataset,
        args.config,
        split=args.split,
        streaming=True,
        data_files=data_files,
        revision=getattr(args, "revision", None),
    )


def _load_cik_map(path: str) -> Dict[str, str]:
    if not path:
        return {}
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            acc = obj.get("accession")
            cik = obj.get("cik")
            if acc and cik:
                out[str(acc)] = str(cik)
    return out


def gate_check(args):
    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    ds = _load_streaming_dataset(args)
    cik_map = _load_cik_map(args.cik_map)

    stats = {}

    def update(cik, token_count, date_str):
        if cik not in stats:
            stats[cik] = {
                "cik": cik,
                "token_count": 0,
                "num_filings": 0,
                "min_date": None,
                "max_date": None,
            }
        s = stats[cik]
        s["token_count"] += token_count
        s["num_filings"] += 1
        if date_str:
            try:
                dt = datetime.fromisoformat(date_str)
                if s["min_date"] is None or dt < s["min_date"]:
                    s["min_date"] = dt
                if s["max_date"] is None or dt > s["max_date"]:
                    s["max_date"] = dt
            except Exception:
                pass

    form_set = set([f.strip().upper() for f in args.form_types.split(",")])

    processed = 0
    kept = 0
    for ex in ds:
        text, cik, form, date = _get_fields(ex, {
            "text": args.text_field,
            "cik": args.cik_field,
            "form": args.form_field,
            "date": args.date_field,
        }, cik_map)
        processed += 1
        if not cik:
            continue
        if form_set and not _form_matches(form, form_set):
            continue
        text = clean_edgar_text(text)
        if not text:
            continue
        token_count = len(tokenizer.encode(text))
        update(cik, token_count, date)
        kept += 1
        if args.log_every and kept % args.log_every == 0:
            print(json.dumps({
                "processed": processed,
                "kept": kept,
                "unique_companies": len(stats),
            }))

    rows = []
    for cik, s in stats.items():
        rows.append({
            "cik": cik,
            "token_count": s["token_count"],
            "num_filings": s["num_filings"],
            "min_date": s["min_date"].date().isoformat() if s["min_date"] else None,
            "max_date": s["max_date"].date().isoformat() if s["max_date"] else None,
        })

    rows = sorted(rows, key=lambda r: r["token_count"], reverse=True)
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "bean_counter_stats.jsonl")
    write_jsonl(out_path, rows)

    # summary
    threshold = args.min_tokens
    num_over = sum(1 for r in rows if r["token_count"] >= threshold)
    summary = {
        "total_companies": len(rows),
        "min_tokens_threshold": threshold,
        "num_companies_over_threshold": num_over,
        "top_5": rows[:5],
    }
    write_json(os.path.join(args.output_dir, "gate_summary.json"), summary)
    print(json.dumps(summary, indent=2))


def _load_stats(stats_path: str) -> List[Dict]:
    rows = []
    with open(stats_path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rows = sorted(rows, key=lambda r: r["token_count"], reverse=True)
    return rows


def _select_companies(rows: List[Dict], top_n: int, seed: int, n_targets: int) -> Tuple[List[str], List[str]]:
    top = [r["cik"] for r in rows[:top_n]]
    rng = np.random.RandomState(seed)
    rng.shuffle(top)
    targets = sorted(top[:n_targets])
    retain = sorted(top[n_targets:])
    return targets, retain


def _collect_docs(args, selected_ciks: set, include_forms: set, cik_map: Dict[str, str]) -> Dict[str, List[Dict]]:
    ds = _load_streaming_dataset(args)
    docs = defaultdict(list)

    for ex in ds:
        text, cik, form, date = _get_fields(ex, {
            "text": args.text_field,
            "cik": args.cik_field,
            "form": args.form_field,
            "date": args.date_field,
        }, cik_map)
        accession = ex.get("accession")
        if not cik or cik not in selected_ciks:
            continue
        if include_forms and not _form_matches(form, include_forms):
            continue
        text = clean_edgar_text(text)
        if not text:
            continue
        docs[cik].append({
            "cik": cik,
            "text": text,
            "form": form,
            "date": date,
            "accession": accession,
        })
    return docs


def _sample_nonmember_docs(
    args,
    excluded_ciks: set,
    include_forms: set,
    target_token_budget: int,
    max_docs: int,
    cik_map: Dict[str, str],
    excluded_accessions: set = None,
) -> List[Dict]:
    ds = _load_streaming_dataset(args)
    out = []
    token_budget = target_token_budget
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    buffer = []
    buffer_size = getattr(args, "sample_buffer", 0)
    excluded_accessions = excluded_accessions or set()
    rng = np.random.RandomState(getattr(args, "seed", 13))
    for ex in ds:
        text, cik, form, date = _get_fields(ex, {
            "text": args.text_field,
            "cik": args.cik_field,
            "form": args.form_field,
            "date": args.date_field,
        }, cik_map)
        accession = ex.get("accession")
        if not cik or cik in excluded_ciks:
            continue
        if accession and accession in excluded_accessions:
            continue
        if include_forms and not _form_matches(form, include_forms):
            continue
        text = clean_edgar_text(text)
        if not text:
            continue
        tok = len(tokenizer.encode(text))
        item = {"cik": cik, "text": text, "form": form, "date": date, "accession": accession, "_tok": tok}
        if buffer_size and buffer_size > 0:
            buffer.append(item)
            if len(buffer) < buffer_size:
                continue
            rng.shuffle(buffer)
            while buffer:
                it = buffer.pop()
                out.append({k: v for k, v in it.items() if k != "_tok"})
                token_budget -= it["_tok"]
                if (target_token_budget and token_budget <= 0) or (max_docs and len(out) >= max_docs):
                    return out
        else:
            out.append({k: v for k, v in item.items() if k != "_tok"})
            token_budget -= tok
            if (target_token_budget and token_budget <= 0) or (max_docs and len(out) >= max_docs):
                break
    if buffer:
        rng.shuffle(buffer)
        for it in buffer:
            out.append({k: v for k, v in it.items() if k != "_tok"})
            token_budget -= it["_tok"]
            if (target_token_budget and token_budget <= 0) or (max_docs and len(out) >= max_docs):
                break
    return out


def _tokenize_and_chunk(tokenizer, texts: List[str], max_length: int) -> List[Dict]:
    out = []
    for t in texts:
        enc = tokenizer(
            t,
            truncation=True,
            max_length=max_length,
            return_overflowing_tokens=True,
            stride=0,
            padding=False,
        )
        for i in range(len(enc["input_ids"])):
            out.append({
                "input_ids": enc["input_ids"][i],
                "attention_mask": enc["attention_mask"][i],
            })
    return out


def build_splits(args):
    set_seed(args.seed)
    rows = _load_stats(args.stats_path)

    targets, retain = _select_companies(rows, args.top_n, args.seed, args.n_targets)
    selected = set(targets + retain)

    include_forms = set([f.strip().upper() for f in args.form_types.split(",")])

    cik_map = _load_cik_map(args.cik_map)
    docs_by_cik = _collect_docs(args, selected, include_forms, cik_map)

    # dedupe per company (preserve first seen metadata)
    for cik in list(docs_by_cik.keys()):
        seen = set()
        deduped_docs = []
        for d in docs_by_cik[cik]:
            t = d.get("text", "")
            h = stable_hash(t)
            if h in seen:
                continue
            seen.add(h)
            deduped_docs.append(d)
        docs_by_cik[cik] = deduped_docs

    holdout_map = {}
    target_holdout_full = []
    train_docs_c1 = []
    train_docs_retain = []

    rng = np.random.RandomState(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)

    for cik, docs in docs_by_cik.items():
        n = len(docs)
        if n == 0:
            continue
        holdout_n = min(args.holdout_max, max(1, int(args.holdout_frac * n)))
        idx = np.arange(n)
        rng.shuffle(idx)
        holdout_idx = set(idx[:holdout_n].tolist())
        holdout = [docs[i] for i in holdout_idx]
        train = [docs[i] for i in idx if i not in holdout_idx]
        # randomize training docs to avoid temporal bias before capping
        rng.shuffle(train)
        if args.max_tokens_per_company and args.max_tokens_per_company > 0:
            # cap training tokens per company using tokenized length
            kept = []
            total_tokens = 0
            for d in train:
                tok = len(tokenizer.encode(d["text"]))
                if total_tokens + tok > args.max_tokens_per_company:
                    continue
                kept.append(d)
                total_tokens += tok
            train = kept
        # Save full holdout docs for evaluation (targets only)
        if cik in targets:
            target_holdout_full.extend(holdout)
        # Keep minimal metadata for bookkeeping
        holdout_map[cik] = [
            {"accession": d.get("accession"), "form": d.get("form"), "date": d.get("date")}
            for d in holdout
        ]
        if cik in targets:
            train_docs_c1.extend(train)
        else:
            train_docs_retain.extend(train)

    # background corpus from non-selected companies
    excluded = set(selected)
    background = _sample_nonmember_docs(
        args,
        excluded_ciks=excluded,
        include_forms=include_forms,
        target_token_budget=args.background_tokens,
        max_docs=args.background_max_docs,
        cik_map=cik_map,
    )
    background_accessions = {d.get("accession") for d in background if d.get("accession")}
    background_ciks = {d.get("cik") for d in background if d.get("cik")}

    # non-member eval pool
    nonmember_eval = _sample_nonmember_docs(
        args,
        excluded_ciks=excluded.union(background_ciks),
        include_forms=include_forms,
        target_token_budget=args.nonmember_tokens,
        max_docs=args.nonmember_max_docs,
        cik_map=cik_map,
        excluded_accessions=background_accessions,
    )

    def to_dataset(docs: List[Dict]) -> Dataset:
        texts = [d["text"] for d in docs]
        chunks = _tokenize_and_chunk(tokenizer, texts, args.max_length)
        return Dataset.from_list(chunks)

    # teacher training datasets include background corpus
    teacher_c1 = to_dataset(train_docs_c1 + train_docs_retain + background)
    teacher_c2 = to_dataset(train_docs_retain + background)
    # C3 is the clean baseline: background only, with no target or retain docs.
    teacher_c3 = to_dataset(background)

    # distillation training dataset excludes target docs, includes background
    distill = to_dataset(train_docs_retain + background)

    # evaluation datasets
    nonmember_eval_ds = Dataset.from_list(nonmember_eval)
    target_holdout_ds = Dataset.from_list(target_holdout_full)

    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    def _save(ds, path):
        if os.path.exists(path):
            import shutil
            shutil.rmtree(path)
        ds.save_to_disk(path)

    _save(teacher_c1, os.path.join(out_dir, "teacher_c1"))
    _save(teacher_c2, os.path.join(out_dir, "teacher_c2"))
    _save(teacher_c3, os.path.join(out_dir, "teacher_c3"))
    _save(distill, os.path.join(out_dir, "distill"))
    _save(target_holdout_ds, os.path.join(out_dir, "eval_target_holdout"))
    _save(nonmember_eval_ds, os.path.join(out_dir, "eval_nonmember"))

    def _jsonify(obj):
        if isinstance(obj, (datetime, )):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: _jsonify(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_jsonify(v) for v in obj]
        return obj

    split_info = {
        "targets": targets,
        "retain": retain,
        "holdout_max": args.holdout_max,
        "holdout_frac": args.holdout_frac,
        "background_tokens": args.background_tokens,
        "nonmember_tokens": args.nonmember_tokens,
        "max_tokens_per_company": args.max_tokens_per_company,
        "dataset": args.dataset,
        "config": args.config,
        "revision": getattr(args, "revision", None),
        "sample_buffer": args.sample_buffer,
    }
    write_json(os.path.join(args.output_dir, "splits.json"), split_info)
    write_json(os.path.join(args.output_dir, "holdout_map.json"), _jsonify(holdout_map))

    print(json.dumps({
        "targets": len(targets),
        "retain": len(retain),
        "teacher_c1_sequences": len(teacher_c1),
        "teacher_c2_sequences": len(teacher_c2),
        "distill_sequences": len(distill),
        "target_holdout_docs": len(target_holdout_full),
        "nonmember_eval_docs": len(nonmember_eval),
    }, indent=2))


def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    gate = sub.add_parser("gate")
    gate.add_argument("--dataset", default="bradfordlevy/BeanCounter")
    gate.add_argument("--config", default="clean")
    gate.add_argument("--revision", default=None, help="Optional dataset revision/commit")
    gate.add_argument("--split", default="train")
    gate.add_argument("--data-files", default=None, help="Optional: comma-separated, JSON list, or glob path")
    gate.add_argument("--cik-map", default=None, help="JSONL mapping with accession->cik")
    gate.add_argument("--log-every", type=int, default=5000, help="Log progress every N kept filings")
    gate.add_argument("--tokenizer", default="EleutherAI/pythia-1.4b")
    gate.add_argument("--form-types", default="10-K")
    gate.add_argument("--text-field", default="text")
    gate.add_argument("--cik-field", default="cik")
    gate.add_argument("--form-field", default="type_filing")
    gate.add_argument("--date-field", default="date")
    gate.add_argument("--min-tokens", type=int, default=200_000)
    gate.add_argument("--seed", type=int, default=13)
    gate.add_argument("--output-dir", default="data")

    build = sub.add_parser("build")
    build.add_argument("--dataset", default="bradfordlevy/BeanCounter")
    build.add_argument("--config", default="clean")
    build.add_argument("--revision", default=None, help="Optional dataset revision/commit")
    build.add_argument("--split", default="train")
    build.add_argument("--data-files", default=None, help="Optional: comma-separated, JSON list, or glob path")
    build.add_argument("--cik-map", default=None, help="JSONL mapping with accession->cik")
    build.add_argument("--tokenizer", default="EleutherAI/pythia-1.4b")
    build.add_argument("--form-types", default="10-K")
    build.add_argument("--text-field", default="text")
    build.add_argument("--cik-field", default="cik")
    build.add_argument("--form-field", default="type_filing")
    build.add_argument("--date-field", default="date")
    build.add_argument("--stats-path", default="data/bean_counter_stats.jsonl")
    build.add_argument("--output-dir", default="data/datasets")
    build.add_argument("--top-n", type=int, default=100)
    build.add_argument("--n-targets", type=int, default=50)
    build.add_argument("--holdout-max", type=int, default=5)
    build.add_argument("--holdout-frac", type=float, default=0.2)
    build.add_argument("--background-tokens", type=int, default=50_000_000)
    build.add_argument("--background-max-docs", type=int, default=0)
    build.add_argument("--nonmember-tokens", type=int, default=5_000_000)
    build.add_argument("--nonmember-max-docs", type=int, default=0)
    build.add_argument("--sample-buffer", type=int, default=0, help="Shuffle buffer size for sampling nonmembers/background")
    build.add_argument("--max-length", type=int, default=512)
    build.add_argument("--seed", type=int, default=13)
    build.add_argument("--max-tokens-per-company", type=int, default=0, help="0 means no cap")

    return p


def main():
    args = build_parser().parse_args()
    if args.cmd == "gate":
        gate_check(args)
    elif args.cmd == "build":
        build_splits(args)


if __name__ == "__main__":
    main()
