#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-chars", type=int, default=32768)
    parser.add_argument("--bf16", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    model_path = Path(args.model)
    dataset_path = Path(args.dataset)
    output_path = Path(args.output)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.bf16 and device == "cuda" and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(str(model_path)).to(
        device=device,
        dtype=dtype if device == "cuda" else torch.float32,
    )
    model.eval()

    ds = datasets.load_from_disk(str(dataset_path))
    items = [(str(ex.get("cik", "")), ex.get("text", "")) for ex in ds]
    items = [(cik, text) for cik, text in items if cik and text]

    grouped = {}
    with torch.no_grad():
        for i in range(0, len(items), args.batch_size):
            batch = items[i : i + args.batch_size]
            ciks = [cik for cik, _ in batch]
            texts = [text[: args.max_chars] for _, text in batch]
            enc = tokenizer(
                texts,
                truncation=True,
                max_length=args.max_length,
                padding=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_mask = attention_mask[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="none",
            ).view(shift_labels.size())
            loss = (loss * shift_mask).sum(dim=1) / shift_mask.sum(dim=1)
            for cik, value in zip(ciks, loss.detach().cpu().tolist()):
                grouped.setdefault(cik, []).append(float(value))

    result = {
        "dataset": str(dataset_path),
        "model": str(model_path),
        "num_companies": len(grouped),
        "per_company": {
            cik: {"num_docs": len(losses), "mean_loss": float(np.mean(losses))}
            for cik, losses in sorted(grouped.items())
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output_path), "num_companies": len(grouped)}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
