# Artifact Notes

## Condition Definitions

- `C1`: clean training baseline
- `C2`: exact remediation / full retraining excluding the deletion targets
- `C3`: never-trained baseline
- `C6`: NPO-based unlearning from the `C1` initialization using `C1` as the frozen reference policy

The repo should not describe `C2 vs C3` as the sole primary comparison. The core audit compares a deployed `C6` student against a `C1` student reference, with deletion targets as positives and retained training members as negatives.

## Placebo Semantics

The random-forget placebo built by `src/build_random_forget.py` samples candidate companies after excluding both `splits["targets"]` and `splits["retain"]`. In other words, the placebo forget set is disjoint from both the audit positives and the audit negatives.

This makes the current placebo a disjoint wrong-target control rather than a control that perturbs the retained audit negatives directly.

## Portable vs Environment-Specific Entry Points

Portable entry points for reproducing the main experiments are:

- `scripts/reproduce_pythia14_headline.sh`
- `scripts/reproduce_neo13_robustness.sh`
- `scripts/run_c6_npo.sh`
- `scripts/run_c6_npo_neo_1p3b.sh`
- `scripts/make_tables.py`

Environment-specific cluster / staging wrappers are isolated under `scripts/ops/`. They are operational helpers rather than the minimal portable reproduction path, and they now use placeholder `REMOTE_HOST` / `REMOTE_ROOT` defaults that must be overridden locally.
