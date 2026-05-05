import argparse
import json
import math
import os
import random

import datasets
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, help="Path to dataset saved with save_to_disk (tokenized)")
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--max-samples", type=int, default=0, help="If >0, sample this many examples")
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--text-column", default="text")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--device", default="cuda")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    ds = datasets.load_from_disk(args.dataset)
    if args.max_samples and args.max_samples > 0 and len(ds) > args.max_samples:
        rng = random.Random(args.seed)
        idx = list(range(len(ds)))
        rng.shuffle(idx)
        ds = ds.select(idx[: args.max_samples])

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_length

    if "input_ids" not in ds.column_names:
        if args.text_column not in ds.column_names:
            raise ValueError(
                f"Dataset at {args.dataset} is not tokenized and lacks text column '{args.text_column}'. "
                f"Columns: {ds.column_names}"
            )

        def tokenize_batch(batch):
            return tokenizer(
                batch[args.text_column],
                truncation=True,
                max_length=args.max_length,
            )

        ds = ds.map(
            tokenize_batch,
            batched=True,
            remove_columns=ds.column_names,
            desc="Tokenizing raw-text dataset for eval_ppl",
        )

    model = AutoModelForCausalLM.from_pretrained(args.model)
    if args.bf16:
        model = model.to(dtype=torch.bfloat16)
    if args.fp16:
        model = model.to(dtype=torch.float16)
    model = model.to(args.device)
    model.eval()

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collator)

    total_loss = 0.0
    total_tokens = 0
    total_examples = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(args.device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(args.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = out.loss
            if attention_mask is not None:
                tokens = int(attention_mask.sum().item())
            else:
                tokens = int(input_ids.numel())
            total_loss += float(loss.item()) * tokens
            total_tokens += tokens
            total_examples += input_ids.size(0)

    mean_loss = total_loss / max(1, total_tokens)
    ppl = math.exp(mean_loss) if mean_loss < 20 else float("inf")

    result = {
        "mean_loss": float(mean_loss),
        "perplexity": float(ppl),
        "num_tokens": int(total_tokens),
        "num_examples": int(total_examples),
        "dataset": args.dataset,
        "max_samples": args.max_samples,
        "seed": args.seed,
    }

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
