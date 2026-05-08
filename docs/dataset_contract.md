# Dataset Contract

This pipeline can either build datasets from a compatible corpus or run directly on user-supplied Hugging Face `save_to_disk` datasets.

## Minimum prebuilt layout

Under `DATASETS_DIR`, the training and evaluation scripts expect:

- `teacher_c1/`
- `teacher_c2/`
- `teacher_c3/`
- `target_train/`
- `distill/`
- `eval_target_holdout/`
- `eval_nonmember/`
- `holdout_map.json`
- `splits.json`

Optional but required for retain-pool deletion audits:

- external or local `RETAIN_HOLDOUT_DATASET/`

## Field expectations

Training datasets consumed by `train_teacher.py`, `distill_student.py`, and `unlearn_teacher.py` must be valid Hugging Face datasets saved with `save_to_disk`. They should contain tokenizable text under a field named `text`, unless you build them through `src/data_prep.py`.

Evaluation datasets consumed by:

- `src/eval_mia.py`
- `scripts/eval_company_losses.py`

must include:

- `text`: document text
- `cik`: company/entity identifier for per-company aggregation

`holdout_map.json` should map target entities to held-out document identifiers in the format produced by `src/data_prep.py`.

## Builder assumptions

If you use `src/data_prep.py gate/build`, your raw corpus must expose enough metadata to recover:

- entity identifier (`cik` or equivalent)
- filing/document type
- document date
- document text

If your corpus does not match that contract, build the datasets yourself and start the pipeline from the training stage.
