import argparse
import math
import os
import functools
import random
from typing import Dict

import datasets
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, DataCollatorWithPadding

try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        CPUOffload,
        MixedPrecision,
        ShardingStrategy,
        StateDictType,
        FullStateDictConfig,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
    _HAS_FSDP = True
except Exception:
    _HAS_FSDP = False

try:
    import bitsandbytes as bnb
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False


def _move_to_device(batch: Dict[str, torch.Tensor], device: str) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}

def _make_labels(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    labels = batch["input_ids"].clone()
    if "attention_mask" in batch:
        labels[batch["attention_mask"] == 0] = -100
    return labels


def _cycle(dl):
    while True:
        for x in dl:
            yield x


@record
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
    p.add_argument("--kl-model", default=None, help="Reference model for KL regularization")
    p.add_argument("--kl-weight", type=float, default=0.0)
    p.add_argument("--kl-device", default="cpu")
    p.add_argument("--kl-every", type=int, default=1)
    p.add_argument("--early-stop-patience", type=int, default=0)
    p.add_argument("--early-stop-delta", type=float, default=0.0)
    p.add_argument("--early-stop-min-steps", type=int, default=-1, help="If <0, use 10% of total steps")
    p.add_argument("--retain-val-frac", type=float, default=0.05)
    p.add_argument("--retain-val-max", type=int, default=512)
    p.add_argument("--early-stop-eval-every", type=int, default=50)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--optim", default="adamw_8bit", choices=["adamw", "adamw_8bit"])
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--fsdp", action="store_true")
    p.add_argument("--fsdp-layer-cls", default=None, help="Transformer block class for FSDP auto-wrapping, e.g. GPTNeoBlock")
    p.add_argument("--cpu-offload", action="store_true", help="Enable FSDP CPU param offload")
    args = p.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    forget_ds = datasets.load_from_disk(args.forget_dataset)
    retain_ds = datasets.load_from_disk(args.retain_dataset) if args.retain_dataset else None

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1

    if args.fsdp and not _HAS_FSDP:
        raise SystemExit("FSDP requested but not available in this torch build.")

    if is_distributed:
        torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
        if args.fsdp and args.kl_device in ("cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"):
            # Ensure KL batches land on the local rank device under FSDP.
            args.kl_device = args.device

    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.bf16:
        model = model.to(dtype=torch.bfloat16)
    if args.fp16:
        model = model.to(dtype=torch.float16)
    model = model.to(args.device)

    auto_wrap_policy = None
    mixed_precision = None
    cpu_offload = CPUOffload(offload_params=True) if args.cpu_offload else None
    if args.fsdp:
        def _resolve_layer_cls():
            if args.fsdp_layer_cls:
                name = args.fsdp_layer_cls
                # Allow short names
                if name == "GPTNeoBlock":
                    from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock
                    return GPTNeoBlock
                if name == "GPTNeoXLayer":
                    from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
                    return GPTNeoXLayer
                if name == "LlamaDecoderLayer":
                    from transformers.models.llama.modeling_llama import LlamaDecoderLayer
                    return LlamaDecoderLayer
                if name == "GPT2Block":
                    from transformers.models.gpt2.modeling_gpt2 import GPT2Block
                    return GPT2Block
                # Fully-qualified import
                module_path, cls_name = name.rsplit(".", 1)
                mod = __import__(module_path, fromlist=[cls_name])
                return getattr(mod, cls_name)
            # Infer from model type
            mtype = getattr(model.config, "model_type", "")
            if mtype == "gpt_neo":
                from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock
                return GPTNeoBlock
            if mtype == "gpt_neox":
                from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
                return GPTNeoXLayer
            if mtype == "llama":
                from transformers.models.llama.modeling_llama import LlamaDecoderLayer
                return LlamaDecoderLayer
            if mtype == "gpt2":
                from transformers.models.gpt2.modeling_gpt2 import GPT2Block
                return GPT2Block
            # Fallback to GPTNeoXLayer if nothing else matched
            from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer
            return GPTNeoXLayer

        layer_cls = _resolve_layer_cls()
        auto_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={layer_cls},
        )
        mixed_precision = None
        if args.bf16:
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )
        model = FSDP(
            model,
            auto_wrap_policy=auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mixed_precision,
            cpu_offload=cpu_offload,
            device_id=local_rank if is_distributed else None,
        )
    model.train()

    ref_model = None
    if args.kl_model and args.kl_weight > 0:
        ref_model = AutoModelForCausalLM.from_pretrained(args.kl_model)
        ref_model.eval()
        ref_model = ref_model.to(args.kl_device)
        if args.bf16:
            ref_model = ref_model.to(dtype=torch.bfloat16)
        if args.fp16:
            ref_model = ref_model.to(dtype=torch.float16)
        if args.fsdp and is_distributed:
            # Shard KL reference to fit on multi-GPU.
            ref_model = FSDP(
                ref_model,
                auto_wrap_policy=auto_wrap_policy,
                sharding_strategy=ShardingStrategy.FULL_SHARD,
                mixed_precision=mixed_precision,
                cpu_offload=cpu_offload,
                device_id=local_rank,
            )

    collate_fn = DataCollatorWithPadding(tokenizer, return_tensors="pt")
    forget_sampler = DistributedSampler(forget_ds, shuffle=True, seed=args.seed) if is_distributed else None
    forget_loader = DataLoader(
        forget_ds,
        batch_size=args.batch_size,
        shuffle=(forget_sampler is None),
        sampler=forget_sampler,
        collate_fn=collate_fn,
    )
    retain_loader = None
    retain_val_loader = None
    if retain_ds is not None:
        # split retain into train/val for early stopping
        if args.retain_val_frac > 0:
            val_size = min(int(len(retain_ds) * args.retain_val_frac), args.retain_val_max)
            if val_size > 0:
                idx = list(range(len(retain_ds)))
                rng = random.Random(args.seed)
                rng.shuffle(idx)
                val_idx = idx[:val_size]
                train_idx = idx[val_size:]
                retain_val = retain_ds.select(val_idx)
                retain_ds = retain_ds.select(train_idx) if train_idx else retain_ds
                retain_val_loader = DataLoader(
                    retain_val,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=collate_fn,
                )
        retain_sampler = DistributedSampler(retain_ds, shuffle=True, seed=args.seed) if is_distributed else None
        retain_loader = DataLoader(
            retain_ds,
            batch_size=args.batch_size,
            shuffle=(retain_sampler is None),
            sampler=retain_sampler,
            collate_fn=collate_fn,
        )

    if args.cpu_offload and args.optim == "adamw_8bit":
        print("WARNING: --cpu-offload is incompatible with bitsandbytes 8-bit optimizer; switching to AdamW.")
        args.optim = "adamw"

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
    best_retain = None
    bad_count = 0
    if args.early_stop_min_steps < 0:
        args.early_stop_min_steps = max(1, int(0.1 * total_steps))
    while step < total_steps:
        # gradient accumulation
        loss_f_val = None
        loss_r_val = None
        for _ in range(args.grad_accum):
            batch_f = next(forget_iter)
            batch_f = _move_to_device(batch_f, args.device)
            labels_f = _make_labels(batch_f)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                out_f = model(**batch_f, labels=labels_f)
                loss_f = out_f.loss
            loss = -args.alpha * loss_f
            loss_f_val = loss_f.detach().float().item()

            if retain_iter is not None:
                batch_r = next(retain_iter)
                batch_r = _move_to_device(batch_r, args.device)
                labels_r = _make_labels(batch_r)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                    out_r = model(**batch_r, labels=labels_r)
                    loss_r = out_r.loss
                loss += args.beta * loss_r
                loss_r_val = loss_r.detach().float().item()
                # KL regularization to reference model on retain batch
                if ref_model is not None and args.kl_weight > 0 and (step % args.kl_every == 0):
                    with torch.no_grad():
                        ref_batch = {k: v.to(args.kl_device) for k, v in batch_r.items()}
                        ref_out = ref_model(**ref_batch)
                        ref_logits = ref_out.logits.to(args.device)
                    logits = out_r.logits
                    logp = torch.log_softmax(logits, dim=-1)
                    pref = torch.softmax(ref_logits, dim=-1)
                    kl = (pref * (torch.log(pref + 1e-8) - logp)).sum(-1)
                    if "attention_mask" in batch_r:
                        mask = batch_r["attention_mask"].float()
                        kl = (kl * mask).sum() / mask.sum().clamp_min(1.0)
                    else:
                        kl = kl.mean()
                    loss += args.kl_weight * kl

            loss = loss / args.grad_accum
            loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % args.log_every == 0 and (not is_distributed or local_rank == 0):
            msg = {
                "step": step,
                "total_steps": total_steps,
                "forget_loss": loss_f_val,
                "retain_loss": loss_r_val,
            }
            print(msg)

        # Early stopping on retain validation loss (rank0 controls)
        stop = False
        if (
            args.early_stop_patience > 0
            and retain_val_loader is not None
            and step >= args.early_stop_min_steps
            and step % args.early_stop_eval_every == 0
        ):
            model.eval()
            total = 0.0
            count = 0
            with torch.no_grad():
                for batch in retain_val_loader:
                    batch = _move_to_device(batch, args.device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=args.bf16):
                        out = model(**batch, labels=batch["input_ids"])
                        loss_r = out.loss
                    total += float(loss_r.item())
                    count += 1
            val_loss = total / max(1, count)
            if is_distributed:
                t = torch.tensor([val_loss], device=args.device)
                torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.SUM)
                val_loss = float((t / world_size).item())
            model.train()
            if best_retain is None or val_loss < best_retain - args.early_stop_delta:
                best_retain = val_loss
                bad_count = 0
            else:
                bad_count += 1
                if bad_count >= args.early_stop_patience:
                    stop = True

        if is_distributed:
            flag = torch.tensor([1 if stop else 0], device=args.device)
            torch.distributed.broadcast(flag, src=0)
            if flag.item() == 1:
                break
        else:
            if stop:
                break

    if is_distributed:
        torch.distributed.barrier()

    if args.fsdp:
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            cpu_state = model.state_dict()
        if not is_distributed or local_rank == 0:
            os.makedirs(args.output, exist_ok=True)
            model.save_pretrained(args.output, state_dict=cpu_state)
            tokenizer.save_pretrained(args.output)
    else:
        if not is_distributed or local_rank == 0:
            os.makedirs(args.output, exist_ok=True)
            model.save_pretrained(args.output)
            tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
