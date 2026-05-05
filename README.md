# Distillation Propagation Experiment

This repo contains a runnable pipeline for the BeanCounter-based distillation propagation experiment.
It implements dataset gating, teacher fine-tuning, student distillation, and loss-based MIA evaluation.

## Quick Start

1. Create and activate a venv.
2. Install deps: `pip install -r requirements.txt`
3. Run the gate check to verify per-company token volume.
4. Build dataset splits.
5. Train teachers, distill students, and run MIA evaluation.

### Commands

```bash
bash scripts/run_gate.sh
bash scripts/run_build.sh
bash scripts/run_train_teachers.sh
bash scripts/run_distill.sh
bash scripts/run_eval.sh
```

## Structure

- `src/data_prep.py`: dataset loading, cleaning, tokenization, gating, and split creation
- `src/train_teacher.py`: fine-tune teacher models for C1/C2/C3
- `src/distill_student.py`: distill student models from each teacher
- `src/eval_mia.py`: loss-based MIA metrics and AUROC
- `src/utils.py`: shared utilities
- `scripts/*.sh`: example multi-GPU launch scripts

## Expected Outputs

- `data/bean_counter_stats.jsonl`: per-company stats
- `data/splits/*.json`: target/retain lists and holdout mapping
- `data/datasets/*`: processed datasets for training and evaluation
- `outputs/teachers/*`: teacher checkpoints
- `outputs/students/*`: student checkpoints
- `outputs/mia/*.json`: per-condition MIA results

## GPU Usage

Use `torchrun --nproc_per_node 4` on a 4x3090 machine. Example scripts are in `scripts/`.

## Notes

- Distillation data excludes target companies in all conditions.
- The student never sees target filings directly.
- The C2 vs C3 comparison is the primary test.
- BeanCounter configs available: `default`, `clean`, `sample`, `fraud`. Scripts use `clean`.
- BeanCounter does not expose a CIK column; we build a mapping from SEC master index files.
- If needed, you can pass `--data-files` to target specific shards (comma-separated, JSON list, or glob).
- Run `scripts/run_sec_index.sh` before gating/building to create `data/sec_index_10k.jsonl`.
- To enable 10-Q fallback, set `FALLBACK_10Q=1` when running `scripts/run_gate.sh` and `scripts/run_build.sh` (default is 10-K only).
- Non-member eval pool is set to 15M tokens in `scripts/run_build.sh`.
- Teacher training runs one GPU per condition in parallel. Use `OPTIM=adamw_8bit` to enable 8-bit AdamW (requires `bitsandbytes`).
- You can override batch/accum via `BATCH=4 ACCUM=8` (defaults are 4/8 for effective batch 32).
- To cap per-company training volume, set `MAX_TOKENS_PER_COMPANY` when running `scripts/run_build.sh` (e.g. `500000`).
- Per-company cap is applied **after** holdout selection, and training docs are shuffled to avoid temporal bias.

## Reproducibility Notes

- Run-specific numeric results are not tracked in this README. Keep results in `outputs/` artifacts or external summary sheets.
- For review-oriented entry points and implementation pointers, see `docs/review_bundle/README.md`.
