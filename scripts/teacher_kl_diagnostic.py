#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--c6-model', required=True)
    p.add_argument('--c1-model', required=True)
    p.add_argument('--c3-model', required=True)
    p.add_argument('--target-dataset', required=True)
    p.add_argument('--retain-dataset', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--batch-size', type=int, default=2)
    p.add_argument('--max-length', type=int, default=512)
    p.add_argument('--max-chars', type=int, default=32768)
    return p


def stats(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        'mean': float(arr.mean()),
        'std': float(arr.std()),
        'median': float(np.median(arr)),
        'n': int(arr.size),
    }


def summarize_pair(c6_model, ref_model, tokenizer, dataset_path, batch_size, max_length, max_chars, device):
    ds = datasets.load_from_disk(str(dataset_path))
    items = [(str(ex.get('cik', '')), ex.get('text', '')) for ex in ds]
    items = [(cik, text) for cik, text in items if cik and text]

    kl_c6_to_ref_vals = []
    kl_ref_to_c6_vals = []
    js_vals = []

    with torch.inference_mode():
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            texts = [text[:max_chars] for _, text in batch]
            enc = tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors='pt',
            )
            input_ids = enc['input_ids'].to(device)
            attention_mask = enc['attention_mask'].to(device)
            mask = attention_mask[:, 1:].contiguous().bool()

            c6_logits = c6_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :].float()
            ref_logits = ref_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :].float()

            c6_log_probs = F.log_softmax(c6_logits, dim=-1)
            ref_log_probs = F.log_softmax(ref_logits, dim=-1)
            c6_probs = c6_log_probs.exp()
            ref_probs = ref_log_probs.exp()
            m_probs = 0.5 * (c6_probs + ref_probs)
            m_log_probs = torch.log(m_probs.clamp_min(1e-12))

            kl_c6_to_ref = (c6_probs * (c6_log_probs - ref_log_probs)).sum(dim=-1)
            kl_ref_to_c6 = (ref_probs * (ref_log_probs - c6_log_probs)).sum(dim=-1)
            js = 0.5 * (
                (c6_probs * (c6_log_probs - m_log_probs)).sum(dim=-1) +
                (ref_probs * (ref_log_probs - m_log_probs)).sum(dim=-1)
            )

            kl_c6_to_ref_vals.extend(kl_c6_to_ref[mask].detach().cpu().tolist())
            kl_ref_to_c6_vals.extend(kl_ref_to_c6[mask].detach().cpu().tolist())
            js_vals.extend(js[mask].detach().cpu().tolist())

            done = min(i + batch_size, len(items))
            if done == len(batch) or done % 40 == 0 or done == len(items):
                print(json.dumps({'dataset': str(dataset_path), 'done': done, 'total': len(items)}), flush=True)

    return {
        'kl_c6_to_ref': stats(kl_c6_to_ref_vals),
        'kl_ref_to_c6': stats(kl_ref_to_c6_vals),
        'js_proxy_mean': float(np.mean(np.asarray(js_vals, dtype=np.float64))),
    }


def main():
    args = build_parser().parse_args()
    c6_model_path = Path(args.c6_model)
    c1_model_path = Path(args.c1_model)
    c3_model_path = Path(args.c3_model)
    target_dataset = Path(args.target_dataset)
    retain_dataset = Path(args.retain_dataset)
    out_path = Path(args.output)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.bfloat16 if device == 'cuda' and torch.cuda.is_bf16_supported() else (torch.float16 if device == 'cuda' else torch.float32)

    tokenizer = AutoTokenizer.from_pretrained(str(c6_model_path), use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    c6_model = AutoModelForCausalLM.from_pretrained(str(c6_model_path)).to(device=device, dtype=dtype)
    c6_model.eval()

    result = {}
    for ref_key, ref_path in [('c1_teacher', c1_model_path), ('c3_teacher', c3_model_path)]:
        ref_model = AutoModelForCausalLM.from_pretrained(str(ref_path)).to(device=device, dtype=dtype)
        ref_model.eval()
        result[ref_key] = {
            'target': summarize_pair(c6_model, ref_model, tokenizer, target_dataset, args.batch_size, args.max_length, args.max_chars, device),
            'retain': summarize_pair(c6_model, ref_model, tokenizer, retain_dataset, args.batch_size, args.max_length, args.max_chars, device),
        }
        del ref_model
        if device == 'cuda':
            torch.cuda.empty_cache()

    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
