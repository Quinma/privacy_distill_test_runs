#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-user@cluster.example.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/path/to/privacy_distill_test_runs}"

ssh "$REMOTE_HOST" "bash -s" <<'REMOTE'
set -euo pipefail

ROOT="${REMOTE_ROOT:-/path/to/privacy_distill_test_runs}"
cd "$ROOT"

ts() { date +%Y%m%d_%H%M%S; }

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: missing directory $path" >&2
    exit 1
  fi
}

require_model_dir() {
  local path="$1"
  if [[ ! -f "$path/model.safetensors" && ! -f "$path/model.safetensors.index.json" && ! -f "$path/pytorch_model.bin" && ! -f "$path/model-00001-of-00002.safetensors" ]]; then
    echo "ERROR: missing model artifacts in $path" >&2
    exit 1
  fi
}

quarantine_path() {
  local path="$1"
  local bucket="$2"
  if [[ -e "$path" ]]; then
    mkdir -p "$bucket"
    mv "$path" "$bucket/"
  fi
}

quarantine_family() {
  local run_tag="$1"
  local bucket="$ROOT/quarantine/c23_repair_$(ts)/$run_tag"
  mkdir -p "$bucket"

  quarantine_path "$ROOT/outputs/$run_tag/teachers/c2" "$bucket"
  quarantine_path "$ROOT/outputs/$run_tag/teachers/c3" "$bucket"
  quarantine_path "$ROOT/outputs/$run_tag/students/c2" "$bucket"
  quarantine_path "$ROOT/outputs/$run_tag/students/c3" "$bucket"

  quarantine_path "$ROOT/outputs/$run_tag/mia/c2_student.json" "$bucket"
  quarantine_path "$ROOT/outputs/$run_tag/mia/c3_student.json" "$bucket"
  quarantine_path "$ROOT/outputs/$run_tag/mia/utility_c2_student.json" "$bucket"
  quarantine_path "$ROOT/outputs/$run_tag/mia/utility_c3_student.json" "$bucket"
  quarantine_path "$ROOT/outputs/$run_tag/mia/utility_c2_student_holdout.json" "$bucket"
  quarantine_path "$ROOT/outputs/$run_tag/mia/utility_c3_student_holdout.json" "$bucket"
  quarantine_path "$ROOT/outputs/$run_tag/mia/stats_bootstrap.json" "$bucket"
  quarantine_path "$ROOT/outputs/$run_tag/mia/summary_table.csv" "$bucket"
  quarantine_path "$ROOT/outputs/$run_tag/mia/summary_table.md" "$bucket"
  quarantine_path "$ROOT/outputs/$run_tag/mia/figure.png" "$bucket"
  quarantine_path "$ROOT/outputs/$run_tag/mia/figure_partial.png" "$bucket"
  quarantine_path "$ROOT/outputs/$run_tag/mia/figure_partial_with_c5_teacher.png" "$bucket"
}

submit_job() {
  local output
  output="$(eval "$*")"
  echo "$output" >&2
  echo "$output" | awk '{print $3}'
}

require_dir "$ROOT/data/datasets/gpt-neo-fixed-20260419/teacher_c1"
require_dir "$ROOT/data/datasets/gpt-neo-fixed-20260419/teacher_c2"
require_dir "$ROOT/data/datasets/gpt-neo-fixed-20260419/teacher_c3"
require_dir "$ROOT/data/datasets/gpt-neo-fixed-20260419/distill"
require_dir "$ROOT/data/datasets/gpt-neo-fixed-20260419/eval_target_holdout"
require_dir "$ROOT/data/datasets/gpt-neo-fixed-20260419/eval_nonmember"

require_dir "$ROOT/data/datasets/pythia-fixed-20260419/teacher_c1"
require_dir "$ROOT/data/datasets/pythia-fixed-20260419/teacher_c2"
require_dir "$ROOT/data/datasets/pythia-fixed-20260419/teacher_c3"
require_dir "$ROOT/data/datasets/pythia-fixed-20260419/distill"
require_dir "$ROOT/data/datasets/pythia-fixed-20260419/eval_target_holdout"
require_dir "$ROOT/data/datasets/pythia-fixed-20260419/eval_nonmember"

require_model_dir "$ROOT/outputs/gpt-neo-1.3b-local/teachers/c1"
require_model_dir "$ROOT/outputs/gpt-neo-1.3b-local/students/c1"
require_model_dir "$ROOT/outputs/gpt-neo-2.7b/teachers/c1"
require_model_dir "$ROOT/outputs/gpt-neo-2.7b/students/c1"
require_model_dir "$ROOT/outputs/pythia-1.4b/teachers/c1"
require_model_dir "$ROOT/outputs/pythia-1.4b/students/c1"
require_model_dir "$ROOT/outputs/pythia-2.8b/teachers/c1"

quarantine_family "gpt-neo-1.3b-local"
quarantine_family "gpt-neo-2.7b"
quarantine_family "pythia-1.4b"
quarantine_family "pythia-2.8b"

neo27_teachers="$(submit_job "qsub -N neo27b-c23-t2 -l h_rt=48:00:00 -l mem=8G -l gpu=4 -pe smp 8 -ac allow=L -v REPO_ROOT=$ROOT,MODEL=EleutherAI/gpt-neo-2.7B,STUDENT=EleutherAI/gpt-neo-125M,RUN_TAG=gpt-neo-2.7b,DATASETS_DIR=$ROOT/data/datasets/gpt-neo-fixed-20260419,REUSE_DATASETS=1,RUN_C1=0,RUN_C2=1,RUN_C3=1,TEACHER_ONLY=1,TRAIN_FSDP=1,TRAIN_FSDP_NPROC=4,TRAIN_GRAD_CHECKPOINTING=1,DISTILL_DDP_NPROC=4,TRAIN_DDP_NPROC=4,UNLEARN_FSDP_NPROC=4,RUN_CMD=./scripts/run_pipeline_full.sh cluster/qsub_run.sh")"
submit_job "qsub -N neo27b-c23-s2 -hold_jid $neo27_teachers -l h_rt=36:00:00 -l mem=8G -l gpu=4 -pe smp 8 -ac allow=L -v REPO_ROOT=$ROOT,MODEL=EleutherAI/gpt-neo-2.7B,STUDENT=EleutherAI/gpt-neo-125M,RUN_TAG=gpt-neo-2.7b,DATASETS_DIR=$ROOT/data/datasets/gpt-neo-fixed-20260419,REUSE_DATASETS=1,RUN_C1=0,RUN_C2=1,RUN_C3=1,TEACHER_ONLY=0,RUN_UTILITY=1,TRAIN_FSDP=1,TRAIN_FSDP_NPROC=4,TRAIN_GRAD_CHECKPOINTING=1,DISTILL_DDP_NPROC=4,TRAIN_DDP_NPROC=4,UNLEARN_FSDP_NPROC=4,RUN_CMD=./scripts/run_pipeline_full.sh cluster/qsub_run.sh" >/dev/null

neo13_teachers="$(submit_job "qsub -N neo13b-c23-t2 -l h_rt=36:00:00 -l mem=8G -l gpu=4 -pe smp 8 -ac allow=L -v REPO_ROOT=$ROOT,MODEL=EleutherAI/gpt-neo-1.3B,STUDENT=EleutherAI/gpt-neo-125M,RUN_TAG=gpt-neo-1.3b-local,DATASETS_DIR=$ROOT/data/datasets/gpt-neo-fixed-20260419,REUSE_DATASETS=1,RUN_C1=0,RUN_C2=1,RUN_C3=1,TEACHER_ONLY=1,TRAIN_FSDP=1,TRAIN_FSDP_NPROC=4,TRAIN_GRAD_CHECKPOINTING=1,DISTILL_DDP_NPROC=4,TRAIN_DDP_NPROC=4,UNLEARN_FSDP_NPROC=4,RUN_CMD=./scripts/run_pipeline_full.sh cluster/qsub_run.sh")"
submit_job "qsub -N neo13b-c23-s2 -hold_jid $neo13_teachers -l h_rt=24:00:00 -l mem=8G -l gpu=4 -pe smp 8 -ac allow=L -v REPO_ROOT=$ROOT,MODEL=EleutherAI/gpt-neo-1.3B,STUDENT=EleutherAI/gpt-neo-125M,RUN_TAG=gpt-neo-1.3b-local,DATASETS_DIR=$ROOT/data/datasets/gpt-neo-fixed-20260419,REUSE_DATASETS=1,RUN_C1=0,RUN_C2=1,RUN_C3=1,TEACHER_ONLY=0,RUN_UTILITY=1,TRAIN_FSDP=1,TRAIN_FSDP_NPROC=4,TRAIN_GRAD_CHECKPOINTING=1,DISTILL_DDP_NPROC=4,TRAIN_DDP_NPROC=4,UNLEARN_FSDP_NPROC=4,RUN_CMD=./scripts/run_pipeline_full.sh cluster/qsub_run.sh" >/dev/null

py14_teachers="$(submit_job "qsub -N p14-c23-t2 -l h_rt=36:00:00 -l mem=8G -l gpu=4 -pe smp 8 -ac allow=L -v REPO_ROOT=$ROOT,MODEL=EleutherAI/pythia-1.4b,STUDENT=EleutherAI/pythia-410m,RUN_TAG=pythia-1.4b,DATASETS_DIR=$ROOT/data/datasets/pythia-fixed-20260419,REUSE_DATASETS=1,RUN_C1=0,RUN_C2=1,RUN_C3=1,TEACHER_ONLY=1,TRAIN_FSDP=1,TRAIN_FSDP_NPROC=4,TRAIN_GRAD_CHECKPOINTING=1,DISTILL_DDP_NPROC=4,TRAIN_DDP_NPROC=4,UNLEARN_FSDP_NPROC=4,RUN_CMD=./scripts/run_pipeline_full.sh cluster/qsub_run.sh")"
submit_job "qsub -N p14-c23-s2 -hold_jid $py14_teachers -l h_rt=24:00:00 -l mem=8G -l gpu=4 -pe smp 8 -ac allow=L -v REPO_ROOT=$ROOT,MODEL=EleutherAI/pythia-1.4b,STUDENT=EleutherAI/pythia-410m,RUN_TAG=pythia-1.4b,DATASETS_DIR=$ROOT/data/datasets/pythia-fixed-20260419,REUSE_DATASETS=1,RUN_C1=0,RUN_C2=1,RUN_C3=1,TEACHER_ONLY=0,RUN_UTILITY=1,TRAIN_FSDP=1,TRAIN_FSDP_NPROC=4,TRAIN_GRAD_CHECKPOINTING=1,DISTILL_DDP_NPROC=4,TRAIN_DDP_NPROC=4,UNLEARN_FSDP_NPROC=4,RUN_CMD=./scripts/run_pipeline_full.sh cluster/qsub_run.sh" >/dev/null

py28_teachers="$(submit_job "qsub -N p28-c23-t2 -l h_rt=48:00:00 -l mem=8G -l gpu=4 -pe smp 8 -ac allow=L -v REPO_ROOT=$ROOT,MODEL=EleutherAI/pythia-2.8b,STUDENT=EleutherAI/pythia-410m,RUN_TAG=pythia-2.8b,DATASETS_DIR=$ROOT/data/datasets/pythia-fixed-20260419,REUSE_DATASETS=1,RUN_C1=0,RUN_C2=1,RUN_C3=1,TEACHER_ONLY=1,TRAIN_FSDP=1,TRAIN_FSDP_NPROC=4,TRAIN_GRAD_CHECKPOINTING=1,DISTILL_DDP_NPROC=4,TRAIN_DDP_NPROC=4,UNLEARN_FSDP_NPROC=4,RUN_CMD=./scripts/run_pipeline_full.sh cluster/qsub_run.sh")"
submit_job "qsub -N p28-c23-s2 -hold_jid $py28_teachers -l h_rt=36:00:00 -l mem=8G -l gpu=4 -pe smp 8 -ac allow=L -v REPO_ROOT=$ROOT,MODEL=EleutherAI/pythia-2.8b,STUDENT=EleutherAI/pythia-410m,RUN_TAG=pythia-2.8b,DATASETS_DIR=$ROOT/data/datasets/pythia-fixed-20260419,REUSE_DATASETS=1,RUN_C1=1,RUN_C2=1,RUN_C3=1,TEACHER_ONLY=0,RUN_UTILITY=1,TRAIN_FSDP=1,TRAIN_FSDP_NPROC=4,TRAIN_GRAD_CHECKPOINTING=1,DISTILL_DDP_NPROC=4,TRAIN_DDP_NPROC=4,UNLEARN_FSDP_NPROC=4,RUN_CMD=./scripts/run_pipeline_full.sh cluster/qsub_run.sh" >/dev/null

echo "Submitted repair chains:"
echo "  neo27b teacher job: $neo27_teachers"
echo "  neo13b teacher job: $neo13_teachers"
echo "  p14 teacher job:    $py14_teachers"
echo "  p28 teacher job:    $py28_teachers"
echo
echo "Use: qstat -u \$USER"
REMOTE
