import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer

from data_prep import _get_fields, _load_cik_map, _load_streaming_dataset, _form_matches
from utils import clean_edgar_text, set_seed, stable_hash, write_json


def _jsonify(obj):
    if isinstance(obj, (datetime,)):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    return obj


def _load_stats(stats_path: str) -> List[Dict]:
    rows = []
    with open(stats_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
    rows = sorted(rows, key=lambda r: r["token_count"], reverse=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="bradfordlevy/BeanCounter")
    p.add_argument("--config", default="clean")
    p.add_argument("--revision", default=None, help="Optional dataset revision/commit")
    p.add_argument("--split", default="train")
    p.add_argument("--data-files", default=None, help="Optional: comma-separated, JSON list, or glob path")
    p.add_argument("--cik-map", required=True)
    p.add_argument("--tokenizer", default="EleutherAI/pythia-1.4b")
    p.add_argument("--form-types", default="10-K")
    p.add_argument("--text-field", default="text")
    p.add_argument("--cik-field", default="cik")
    p.add_argument("--form-field", default="type_filing")
    p.add_argument("--date-field", default="date")
    p.add_argument("--splits", required=True)
    p.add_argument("--stats-path", required=True)
    p.add_argument("--num-forget", type=int, default=50)
    p.add_argument("--min-tokens", type=int, default=200000)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--max-tokens-per-company", type=int, default=0)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--output", default="data/datasets/random_forget_train")
    p.add_argument("--ciks-output", default=None, help="Optional JSON file to store selected CIKs")
    args = p.parse_args()

    set_seed(args.seed)

    with open(args.splits, "r", encoding="utf-8") as f:
        splits = json.load(f)
    if args.max_tokens_per_company == 0 and splits.get("max_tokens_per_company"):
        args.max_tokens_per_company = int(splits["max_tokens_per_company"])

    excluded = set(splits.get("targets", [])) | set(splits.get("retain", []))

    stats = _load_stats(args.stats_path)
    candidates = [r for r in stats if r.get("token_count", 0) >= args.min_tokens and r.get("cik") not in excluded]
    if len(candidates) < args.num_forget:
        raise SystemExit(f"Not enough candidate companies for random forget set: {len(candidates)} < {args.num_forget}")

    rng = np.random.RandomState(args.seed)
    idx = np.arange(len(candidates))
    rng.shuffle(idx)
    chosen = [candidates[i]["cik"] for i in idx[: args.num_forget]]
    chosen = sorted(chosen)

    if args.ciks_output:
        os.makedirs(os.path.dirname(args.ciks_output), exist_ok=True)
        write_json(args.ciks_output, {"seed": args.seed, "num_forget": args.num_forget, "min_tokens": args.min_tokens, "ciks": chosen})

    include_forms = set([f.strip().upper() for f in args.form_types.split(",") if f.strip()])
    cik_map = _load_cik_map(args.cik_map)

    ds = _load_streaming_dataset(args)
    docs_by_cik: Dict[str, List[Dict]] = defaultdict(list)

    for ex in ds:
        text, cik, form, date = _get_fields(
            ex,
            {
                "text": args.text_field,
                "cik": args.cik_field,
                "form": args.form_field,
                "date": args.date_field,
            },
            cik_map,
        )
        if not cik or cik not in chosen:
            continue
        if include_forms and not _form_matches(form, include_forms):
            continue
        text = clean_edgar_text(text)
        if not text:
            continue
        docs_by_cik[cik].append({
            "cik": cik,
            "text": text,
            "form": form,
            "date": date,
            "accession": ex.get("accession"),
        })

    # dedupe per company
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

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    rng = np.random.RandomState(args.seed)

    train_docs = []
    for cik, docs in docs_by_cik.items():
        if not docs:
            continue
        idx = np.arange(len(docs))
        rng.shuffle(idx)
        shuffled = [docs[i] for i in idx]
        if args.max_tokens_per_company and args.max_tokens_per_company > 0:
            kept = []
            total_tokens = 0
            for d in shuffled:
                tok = len(tokenizer.encode(d["text"]))
                if total_tokens + tok > args.max_tokens_per_company:
                    continue
                kept.append(d)
                total_tokens += tok
            shuffled = kept
        train_docs.extend(shuffled)

    # tokenize and chunk
    texts = [d["text"] for d in train_docs]
    chunks = []
    for t in texts:
        enc = tokenizer(
            t,
            truncation=True,
            max_length=args.max_length,
            return_overflowing_tokens=True,
            stride=0,
            padding=False,
        )
        for i in range(len(enc["input_ids"])):
            chunks.append({
                "input_ids": enc["input_ids"][i],
                "attention_mask": enc["attention_mask"][i],
            })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    if os.path.exists(args.output):
        import shutil
        shutil.rmtree(args.output)
    Dataset.from_list(chunks).save_to_disk(args.output)

    meta = {
        "num_forget": args.num_forget,
        "min_tokens": args.min_tokens,
        "selected_ciks": chosen,
        "train_docs": len(train_docs),
        "train_sequences": len(chunks),
        "max_tokens_per_company": args.max_tokens_per_company,
    }
    write_json(os.path.join(os.path.dirname(args.output), "random_forget_meta.json"), _jsonify(meta))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
