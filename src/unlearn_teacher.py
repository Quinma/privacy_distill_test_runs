import argparse
import math
import os
from typing import Dict

import datasets
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

try:
    import bitsandbytes as bnb
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False


def _move_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _cycle(dl):
    while True:
        for x in dl:
            yield x


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--forget-dataset", required=True)
    p.add_argument("--retain-dataset", default=None)
    p.add_argument("--output", required=True)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--alpha", type=float, default=1.0, help="forget loss weight (ascent)")
    p.add_argument("--beta", type=float, default=1.0, help="retain loss weight (descent)")
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--optim", default="adamw_8bit", choices=["adamw", "adamw_8bit"])
    p.add_argument("--log-every", type=int, default=50)
    args = p.parse_args()

    forget_ds = datasets.load_from_disk(args.forget_dataset)
    retain_ds = datasets.load_from_disk(args.retain_dataset) if args.retain_dataset else None

    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.bf16:
        model = model.to(dtype=torch.bfloat16)
    if args.fp16:
        model = model.to(dtype=torch.float16)
    model = model.to(args.device)
    model.train()

    collate_fn = None
    forget_loader = DataLoader(forget_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    retain_loader = None
    if retain_ds is not None:
        retain_loader = DataLoader(retain_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    if args.optim == "adamw_8bit":
        if not _HAS_BNB:
            raise SystemExit("bitsandbytes is not available; install or use --optim adamw")
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(forget_loader) / max(1, args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * steps_per_epoch
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    forget_iter = _cycle(forget_loader)
    retain_iter = _cycle(retain_loader) if retain_loader is not None else None

    step = 0
    optimizer.zero_grad(set_to_none=True)
    while step < total_steps:
        # gradient accumulation
        loss_f_val = None
        loss_r_val = None
        for _ in range(args.grad_accum):
            batch_f = next(forget_iter)
            batch_f = _move_to_device(batch_f, args.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                out_f = model(**batch_f, labels=batch_f["input_ids"])
                loss_f = out_f.loss
            loss = -args.alpha * loss_f
            loss_f_val = loss_f.detach().float().item()

            if retain_iter is not None:
                batch_r = next(retain_iter)
                batch_r = _move_to_device(batch_r, args.device)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                    out_r = model(**batch_r, labels=batch_r["input_ids"])
                    loss_r = out_r.loss
                loss += args.beta * loss_r
                loss_r_val = loss_r.detach().float().item()

            loss = loss / args.grad_accum
            loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0:
            msg = {
                "step": step,
                "total_steps": total_steps,
                "forget_loss": loss_f_val,
                "retain_loss": loss_r_val,
            }
            print(msg)

    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
