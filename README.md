# NPO Model Audit Pipeline

This repository packages a reproducible audit pipeline for studying how approximate unlearning propagates through a teacher-to-student language-model pipeline.

The public surface is intended for general use:
- you provide the corpus, model choices, and run settings
- the pipeline trains baseline and reference teachers
- distills baseline students
- runs an NPO unlearning branch
- evaluates standard target-vs-nonmember MIA
- optionally runs a retained-member deletion audit over a user-supplied retain holdout

The repository still contains older experiment-specific scripts, but the supported entrypoints for general use are the config-driven wrappers documented below.

## What The Pipeline Covers

Core pipeline stages:
- dataset gate/build for compatible corpora
- baseline teacher training (`C1`, `C2`, `C3`)
- baseline student distillation (`C1`, `C2`, `C3`)
- approximate unlearning with NPO (`C6`)
- student evaluation on target-vs-nonmember holdouts
- optional target-vs-retained-company deletion audit

Useful public scripts:
- `scripts/run_audit_pipeline.sh`
- `scripts/run_retain_audit.sh`
- `scripts/run_gate.sh`
- `scripts/run_build.sh`
- `scripts/run_train_teachers.sh`
- `scripts/run_distill.sh`
- `scripts/run_c6_npo.sh`
- `scripts/run_eval.sh`

## Quick Start

1. Create a virtual environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Copy the example config and edit it for your corpus and models.

```bash
cp configs/audit.env.example configs/my_audit.env
```

3. Decide whether you are:
- supplying prebuilt datasets under `DATASETS_DIR`
- or using the provided builder for a corpus that matches the assumptions in `docs/dataset_contract.md`

4. Run the pipeline.

```bash
bash scripts/run_audit_pipeline.sh configs/my_audit.env
```

## Prebuilt Dataset Mode

If you already have your own datasets, set:
- `RUN_GATE=0`
- `RUN_BUILD=0`

Then place the required datasets and metadata under `DATASETS_DIR` using the contract in `docs/dataset_contract.md`.

This is the recommended path if your corpus does not naturally match the company/accession assumptions in `src/data_prep.py`.

## Raw-Corpus Build Mode

If your raw corpus exposes compatible metadata, you can use:
- `scripts/run_gate.sh`
- `scripts/run_build.sh`

These wrappers are config-driven and accept user-supplied values for:
- dataset name and config
- revision / split / data-files
- tokenizer
- text and metadata field names
- target/retain sampling and token budgets

The build path is optional. It is not required for the training and audit pipeline if you already have prebuilt datasets.

## Dataset Contract

The training and audit steps work on Hugging Face `save_to_disk` datasets.

The minimum public contract is documented in `docs/dataset_contract.md`.

In short, the pipeline expects:
- teacher training datasets (`teacher_c1`, `teacher_c2`, `teacher_c3`)
- distillation dataset (`distill`)
- unlearning forget dataset (`target_train`)
- target holdout dataset (`eval_target_holdout`)
- nonmember dataset (`eval_nonmember`)
- `holdout_map.json`
- optional retain holdout dataset for retained-member auditing

## Retained-Member Audit

To run a retain-pool deletion audit after the standard model pipeline:

```bash
bash scripts/run_retain_audit.sh
```

This wrapper:
- scores `c1`, `c3`, and `c6` students on a retained-member holdout
- aggregates per-company mean losses
- writes `C6-C1` and `C6-C3` target-vs-retain attack JSONs

The retain-holdout dataset is intentionally user-supplied through `RETAIN_HOLDOUT_DATASET`. The repository does not assume a particular corpus provenance for that dataset.

## Outputs

Typical outputs land under:
- `outputs/<RUN_TAG>/teachers/`
- `outputs/<RUN_TAG>/students/`
- `outputs/<RUN_TAG>/mia/`
- `outputs/logs/`

Common result files include:
- `c1_student.json`
- `c2_student.json`
- `c3_student.json`
- `c6_student.json`
- `mia_c6_deletion_attack_target_vs_retain.json`
- `mia_c6_deletion_attack_target_vs_retain_c1ref.json`

## Scope And Limits

This repository is a pipeline, not a benchmark package.

What it does:
- gives you a reproducible training/evaluation skeleton
- makes the audit stages scriptable and restartable
- exposes the unlearning and deletion-audit entrypoints directly

What it does not do:
- choose the right corpus for your domain
- guarantee that `src/data_prep.py` matches arbitrary raw corpora
- provide a universally correct retain-holdout construction for you

If your corpus does not match the build assumptions, prebuild the datasets yourself and use the training/audit stages only.

## Legacy Material

The repository still contains older experiment scripts and environment-specific helpers under `scripts/ops/`. They are not required for the public pipeline above.
