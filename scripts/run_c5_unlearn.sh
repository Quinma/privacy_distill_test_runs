#!/usr/bin/env bash
set -euo pipefail

# Build target-train dataset (targets only, excludes holdout)
python -u src/build_target_train.py \
  --cik-map data/sec_index_10k.jsonl \
  --splits data/datasets/splits.json \
  --holdout-map data/datasets/holdout_map.json \
  --max-tokens-per-company 500000 \
  --output data/datasets/target_train

# Approximate unlearning on C1 teacher
python -u src/unlearn_teacher.py \
  --model outputs/teachers/c1 \
  --forget-dataset data/datasets/target_train \
  --retain-dataset data/datasets/teacher_c2 \
  --output outputs/teachers/c5_unlearn \
  --bf16 \
  --optim adamw_8bit \
  --batch-size 2 \
  --grad-accum 16 \
  --epochs 1 \
  --alpha 1.0 \
  --beta 1.0

# Distill student from unlearned teacher
python -u src/distill_student.py \
  --teacher outputs/teachers/c5_unlearn \
  --student EleutherAI/pythia-410m \
  --train data/datasets/distill \
  --output outputs/students/c5 \
  --bf16 \
  --optim adamw_8bit \
  --batch-size 2 \
  --grad-accum 16 \
  --epochs 1

# Evaluate student on BeanCounter holdout
python -u src/eval_mia.py \
  --model outputs/students/c5 \
  --target-holdout data/datasets/eval_target_holdout \
  --nonmember data/datasets/eval_nonmember \
  --holdout-map data/datasets/holdout_map.json \
  --output outputs/mia/c5_student_bc.json \
  --bf16
