import argparse
import os

import datasets
import torch
import torch.nn.functional as F
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)


class DistillTrainer(Trainer):
    def __init__(self, teacher, temperature=2.0, alpha=0.5, teacher_device="cuda", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_device = torch.device(teacher_device)
        self.teacher.to(self.teacher_device)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")

        outputs = model(**inputs)
        student_logits = outputs.logits

        with torch.no_grad():
            if self.teacher_device != student_logits.device:
                teacher_inputs = {k: v.to(self.teacher_device) for k, v in inputs.items()}
            else:
                teacher_inputs = inputs
            teacher_out = self.teacher(**teacher_inputs)
            teacher_logits = teacher_out.logits
            if teacher_logits.device != student_logits.device:
                teacher_logits = teacher_logits.to(student_logits.device)

        # CE loss (standard LM)
        shift_logits = student_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous() if attention_mask is not None else None

        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        )
        if shift_mask is not None:
            ce_loss = (ce_loss * shift_mask.reshape(-1)).sum() / shift_mask.sum()
        else:
            ce_loss = ce_loss.mean()

        # KD loss
        T = self.temperature
        s_logits = student_logits / T
        t_logits = teacher_logits / T

        s_log_probs = F.log_softmax(s_logits, dim=-1)
        t_probs = F.softmax(t_logits, dim=-1)

        kd_loss = F.kl_div(s_log_probs, t_probs, reduction="none").sum(-1)
        if attention_mask is not None:
            kd_loss = (kd_loss * attention_mask).sum() / attention_mask.sum()
        else:
            kd_loss = kd_loss.mean()
        kd_loss = kd_loss * (T * T)

        loss = self.alpha * ce_loss + (1.0 - self.alpha) * kd_loss

        return (loss, outputs) if return_outputs else loss


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", required=True)
    p.add_argument("--student", default="EleutherAI/pythia-410m")
    p.add_argument("--teacher-device", default="cuda", help="e.g. cuda, cuda:1, cpu")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=None, help="Override epochs with a fixed number of training steps")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--per-device-batch", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=16)
    p.add_argument("--grad-checkpointing", action="store_true")
    p.add_argument("--optim", default="adamw_torch", help="Trainer optimizer name, e.g. adamw_torch or adamw_8bit")
    p.add_argument("--dataloader-num-workers", type=int, default=0)
    p.add_argument("--dataloader-pin-memory", action="store_true")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--save-steps", type=int, default=2000)
    p.add_argument("--logging-steps", type=int, default=50)
    p.add_argument("--resume", default=None)
    return p


def main():
    args = build_parser().parse_args()
    os.makedirs(args.output, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.student, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_length

    ds = datasets.load_from_disk(args.dataset)

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher)
    student = AutoModelForCausalLM.from_pretrained(args.student)
    if args.grad_checkpointing:
        student.gradient_checkpointing_enable()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps and args.max_steps > 0 else -1,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        seed=args.seed,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        gradient_checkpointing=args.grad_checkpointing,
        optim=args.optim,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_pin_memory=args.dataloader_pin_memory,
    )

    trainer = DistillTrainer(
        teacher=teacher,
        temperature=args.temperature,
        alpha=args.alpha,
        teacher_device=args.teacher_device,
        model=student,
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
