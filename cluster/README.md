# Cluster Run Pack (UCL ARC / Slurm + Myriad SGE)

This folder contains the scripts needed to run the full experiment on a Slurm GPU cluster (e.g., Young) or on Myriad (SGE). It includes the full condition set (C1–C5, C5m, C5r), matched-negative evaluation, and seed replications. Defaults are set for GPT‑Neo teachers (1.3B / 2.7B) with GPT‑Neo 125M student.

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

## 2) Load modules and create a venv (Slurm clusters)

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

## 4) Submit the full pipeline (Slurm clusters)

One scale:
```bash
./cluster/submit_scale.sh --model EleutherAI/gpt-neo-1.3B --student EleutherAI/gpt-neo-125M
```

All stages (pipeline + C5 + C5r + matched + seeds):
```bash
./cluster/submit_all.sh --model EleutherAI/gpt-neo-1.3B --student EleutherAI/gpt-neo-125M
```

To run the 2.7B scale:
```bash
./cluster/submit_all.sh --model EleutherAI/gpt-neo-2.7B --student EleutherAI/gpt-neo-125M --run-tag gpt-neo-2.7b
```

To submit both scales automatically:
```bash
./cluster/submit_both_scales.sh
```

## 5) Monitor jobs

```bash
squeue -u $USER
sacct -j <jobid> --format=JobID,JobName,State,Elapsed,MaxRSS
```

## Notes

- The cluster scripts rely on `scripts/run_pipeline_full.sh` plus `scripts/run_c5_aggressive.sh` and `scripts/run_c5r_scale.sh`.
- Matched negatives are handled by `scripts/run_matched_nonmember.sh`, `scripts/run_eval_matched.sh`, and `scripts/run_compute_stats_matched.sh`.
- Seed replications are run by `scripts/run_seed_reps.sh` (scale-aware).

---

# Myriad (SGE) usage

Myriad uses SGE (qsub/qstat) instead of Slurm. Use the `submit_myriad_*.sh` scripts.

## 1) Load modules and create a venv (Myriad)

Edit `cluster/env.sh` to match the Myriad module stack (or create a venv without modules if preferred), then:

```bash
./cluster/setup.sh
```

## 2) Submit the full pipeline (Myriad)

One scale:
```bash
./cluster/submit_myriad_scale.sh --model EleutherAI/gpt-neo-1.3B --student EleutherAI/gpt-neo-125M
```

All stages (pipeline + C5 + C5r + matched + seeds):
```bash
./cluster/submit_myriad_all.sh --model EleutherAI/gpt-neo-1.3B --student EleutherAI/gpt-neo-125M
```

Both scales:
```bash
./cluster/submit_myriad_both_scales.sh
```

Notes:
- `--mem-per-core` is per-core memory (SGE), default `8G`.
- `--allow` can be used to target GPU types (e.g., `L` for A100 40G nodes).

## 3) Monitor jobs (Myriad)

```bash
qstat -u $USER
qacct -j <jobid>
```
