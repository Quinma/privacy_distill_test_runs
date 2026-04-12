import argparse
import os

import datasets
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="EleutherAI/pythia-1.4b")
    p.add_argument("--dataset", required=True, help="Path to dataset saved with save_to_disk")
    p.add_argument("--output", required=True)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--per-device-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--optim", default="adamw_torch", help="Trainer optimizer name, e.g. adamw_torch or adamw_8bit")
    p.add_argument("--fsdp", default=None, help="e.g. 'full_shard auto_wrap'")
    p.add_argument("--fsdp-transformer-layer-cls", default=None, help="e.g. GPTNeoXLayer")
    p.add_argument("--fsdp-min-num-params", type=int, default=0)
    p.add_argument("--dataloader-num-workers", type=int, default=0)
    p.add_argument("--dataloader-pin-memory", action="store_true")
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--save-steps", type=int, default=2000)
    p.add_argument("--save-strategy", default="steps", choices=["no", "steps", "epoch"])
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--eval-steps", type=int, default=0)
    p.add_argument("--resume", default=None)
    return p


@record
def main():
    args = build_parser().parse_args()
    os.makedirs(args.output, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_length

    ds = datasets.load_from_disk(args.dataset)

    model = AutoModelForCausalLM.from_pretrained(args.model)
    if args.grad_checkpointing:
        model.gradient_checkpointing_enable()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        evaluation_strategy="no" if args.eval_steps == 0 else "steps",
        eval_steps=args.eval_steps if args.eval_steps > 0 else None,
        bf16=args.bf16,
        fp16=args.fp16,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=args.grad_checkpointing,
        optim=args.optim,
        fsdp=args.fsdp or "",
        fsdp_transformer_layer_cls_to_wrap=args.fsdp_transformer_layer_cls,
        fsdp_min_num_params=args.fsdp_min_num_params,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
        max_steps=args.max_steps,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == "__main__":
    main()
