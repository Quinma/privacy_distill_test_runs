# Implementation Audit Bundle

This folder indexes the core implementation files that are easiest to miss in a truncated code dump.

Key files:
- `src/distill_student.py`: student KD implementation, including active-token CE/KL masking and teacher-vs-student KL orientation.
- `src/unlearn_teacher.py`: GA and NPO unlearning objective implementation, including frozen reference-model handling for NPO.
- `src/compute_stats.py`: paired bootstrap CI, sign test, and sign-flip permutation test used by the pipeline summary stats.
- `scripts/teacher_feature_attack.py`: teacher-side per-company feature attacks for mean-loss delta, token-KL mean, top-KL subsets, and gold-logit-difference features.
- `scripts/teacher_kl_diagnostic.py`: pooled teacher-side KL / reverse-KL / JS-proxy diagnostics on target and retain pools.

Notes:
- The teacher-feature and teacher-KL computations were previously embedded as Python heredocs inside submission scripts. They are now extracted into standalone Python entry points so the implementation is auditable directly from the repo.
- Run-specific numeric results are intentionally not tracked in the top-level README.

This sanitized branch omits site-specific cluster submission wrappers; the extracted standalone Python entry points remain available for inspection.
