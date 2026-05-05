# Public Release Notes

This repository now separates the portable experiment code from environment-specific operational helpers.

## Portable Artifact Surface

Use these entry points for local reproduction and method inspection:

- `scripts/reproduce_pythia14_headline.sh`
- `scripts/reproduce_neo13_robustness.sh`
- `scripts/run_c6_npo.sh`
- `scripts/run_c6_npo_neo_1p3b.sh`
- `scripts/make_tables.py`
- `src/`
- `docs/implementation_audit/README.md`
- `docs/artifact_notes.md`

## Operational Helpers

Cluster sync / submit / fetch wrappers live under `scripts/ops/`. They are retained as examples from the research workflow, but they are not required for the portable artifact.

These wrappers now use placeholder defaults:

- `REMOTE_HOST=user@cluster.example.edu`
- `REMOTE_ROOT=/path/to/privacy_distill_test_runs`

Set those explicitly in your environment before use.

## Remaining Caveat

The `scripts/ops/` subtree still reflects the original workspace layout and staging conventions. It is suitable as an operational archive, not as the canonical public API for the artifact.
