# Cluster Run Pack (UCL ARC / Slurm)

This folder contains the scripts needed to run the full experiment on a Slurm GPU cluster (e.g., Young). It includes the full condition set (C1–C5, C5m, C5r), matched-negative evaluation, and seed replications.

## 1) Copy the repo to the cluster

Preferred:
```bash
git clone <your_repo_url>
cd <repo>
```

Alternative (from your local machine):
```bash
rsync -av --exclude '.venv' --exclude 'outputs' --exclude 'data' <repo>/ user@cluster:/path/to/repo/
```

## 2) Load modules and create a venv

Edit `cluster/env.sh` to match the cluster stack (Young uses `ucl-stack/2025-12`).

```bash
module purge
module load ucl-stack/2025-12
module load default-modules/2025-12
module load python/3.10.4
module load cuda/12.1
```

Create the venv:
```bash
./cluster/setup.sh
```

## 3) Set credentials and cache locations

```bash
export HF_TOKEN=hf_...
export TRANSFORMERS_CACHE=/path/to/scratch/cache
export HF_HOME=/path/to/scratch/cache
```

## 4) Submit the full pipeline

One scale:
```bash
./cluster/submit_scale.sh --model EleutherAI/pythia-2.8b --student EleutherAI/pythia-410m
```

All stages (pipeline + C5 + C5r + matched + seeds):
```bash
./cluster/submit_all.sh --model EleutherAI/pythia-2.8b --student EleutherAI/pythia-410m
```

## 5) Monitor jobs

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS
```

## Notes

- The cluster scripts rely on `scripts/run_pipeline_full.sh` plus `scripts/run_c5_aggressive.sh` and `scripts/run_c5r_scale.sh`.
- Matched negatives are handled by `scripts/run_matched_nonmember.sh`, `scripts/run_eval_matched.sh`, and `scripts/run_compute_stats_matched.sh`.
- Seed replications are run by `scripts/run_seed_reps_1p4b.sh` (fixed to 1.4B).
