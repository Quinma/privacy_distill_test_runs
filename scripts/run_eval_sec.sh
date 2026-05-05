#!/usr/bin/env bash
set -euo pipefail

# C1 student
python -u src/eval_mia.py \
  --model outputs/students/c1 \
  --target-holdout data/datasets/eval_target_holdout_sec \
  --nonmember data/datasets/eval_nonmember \
  --holdout-map data/datasets/holdout_map.json \
  --output outputs/mia/c1_student_sec.json \
  --bf16

# C2 student
python -u src/eval_mia.py \
  --model outputs/students/c2 \
  --target-holdout data/datasets/eval_target_holdout_sec \
  --nonmember data/datasets/eval_nonmember \
  --holdout-map data/datasets/holdout_map.json \
  --output outputs/mia/c2_student_sec.json \
  --bf16

# C3 student
python -u src/eval_mia.py \
  --model outputs/students/c3 \
  --target-holdout data/datasets/eval_target_holdout_sec \
  --nonmember data/datasets/eval_nonmember \
  --holdout-map data/datasets/holdout_map.json \
  --output outputs/mia/c3_student_sec.json \
  --bf16

# C4 teacher (dirty teacher direct)
python -u src/eval_mia.py \
  --model outputs/teachers/c1 \
  --target-holdout data/datasets/eval_target_holdout_sec \
  --nonmember data/datasets/eval_nonmember \
  --holdout-map data/datasets/holdout_map.json \
  --output outputs/mia/c4_teacher_sec.json \
  --bf16

# Plot
python -u src/plot_results.py \
  --c1 outputs/mia/c1_student_sec.json \
  --c2 outputs/mia/c2_student_sec.json \
  --c3 outputs/mia/c3_student_sec.json \
  --c4 outputs/mia/c4_teacher_sec.json \
  --output outputs/mia/figure_sec.png
