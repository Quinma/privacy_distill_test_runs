#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-user@cluster.example.edu}"
REMOTE_ROOT="${REMOTE_ROOT:-/path/to/privacy_distill_test_runs}"

ssh "$REMOTE_HOST" "REMOTE_ROOT='$REMOTE_ROOT' bash -s" <<'REMOTE'
set -euo pipefail

ROOT="${REMOTE_ROOT:-/path/to/privacy_distill_test_runs}"
cd "$ROOT"

ts() { date +%Y%m%d_%H%M%S; }

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required file: $path" >&2
    exit 1
  fi
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: missing required directory: $path" >&2
    exit 1
  fi
}

require_model_dir() {
  local path="$1"
  if [[ ! -f "$path/model.safetensors" && ! -f "$path/model.safetensors.index.json" && ! -f "$path/pytorch_model.bin" && ! -f "$path/model-00001-of-00002.safetensors" && ! -f "$path/model-00001-of-00003.safetensors" ]]; then
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

quarantine_pythia_students() {
  local run_tag="$1"
  local bucket="$ROOT/quarantine/c23_pythia_student_retry_$(ts)/$run_tag"
  mkdir -p "$bucket"

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
}

submit_job() {
  local output
  output="$(eval "$*")"
  echo "$output" >&2
  echo "$output" | awk '{print $3}'
}

require_file "$ROOT/scripts/run_pipeline_full.sh"
grep -q "SKIP distill_c1 (RUN_C1=0)" "$ROOT/scripts/run_pipeline_full.sh" || {
  echo "ERROR: remote run_pipeline_full.sh does not include RUN_C1 distill skip patch. Run scripts/rsync_c23_retry_patch.sh first." >&2
  exit 1
}
grep -q "DISTILL_TEACHER_DTYPE" "$ROOT/scripts/run_pipeline_full.sh" || {
  echo "ERROR: remote run_pipeline_full.sh does not include DISTILL_TEACHER_DTYPE patch. Run scripts/rsync_c23_retry_patch.sh first." >&2
  exit 1
}

for tag in pythia-1.4b pythia-2.8b; do
  require_dir "$ROOT/data/datasets/pythia-fixed-20260419/distill"
  require_dir "$ROOT/data/datasets/pythia-fixed-20260419/eval_target_holdout"
  require_dir "$ROOT/data/datasets/pythia-fixed-20260419/eval_nonmember"
  require_file "$ROOT/data/datasets/pythia-fixed-20260419/holdout_map.json"
  require_model_dir "$ROOT/outputs/$tag/teachers/c1"
  require_model_dir "$ROOT/outputs/$tag/teachers/c2"
  require_model_dir "$ROOT/outputs/$tag/teachers/c3"
  if [[ ! -f "$ROOT/outputs/$tag/mia/c1_student.json" ]]; then
    echo "ERROR: missing required file: $ROOT/outputs/$tag/mia/c1_student.json" >&2
    echo "Run scripts/rsync_c23_retry_patch.sh first; it now syncs the canonical Pythia C1 MIA baselines needed when RUN_C1=0." >&2
    exit 1
  fi
done

quarantine_pythia_students "pythia-1.4b"
quarantine_pythia_students "pythia-2.8b"

common_safe="DATASETS_DIR=$ROOT/data/datasets/pythia-fixed-20260419,REUSE_DATASETS=1,RUN_C1=0,RUN_C2=1,RUN_C3=1,TEACHER_ONLY=0,RUN_UTILITY=0,BF16=0,OPTIM=adamw_torch,DISTILL_DDP_NPROC=1,DISTILL_TEACHER_DTYPE=float32,DISTILL_TEACHER_DEVICE=cuda,VISIBLE_GPUS=0,TRAIN_DDP_NPROC=1,TRAIN_FSDP_NPROC=1,UNLEARN_FSDP_NPROC=1,RUN_CMD=./scripts/run_pipeline_full.sh"

py14="$(submit_job "qsub -N p14-c23-s3 -l h_rt=24:00:00 -l mem=8G -l gpu=1 -pe smp 4 -ac allow=L -v REPO_ROOT=$ROOT,MODEL=EleutherAI/pythia-1.4b,STUDENT=EleutherAI/pythia-410m,RUN_TAG=pythia-1.4b,$common_safe cluster/qsub_run.sh")"
py28="$(submit_job "qsub -N p28-c23-s3 -l h_rt=30:00:00 -l mem=8G -l gpu=1 -pe smp 4 -ac allow=L -v REPO_ROOT=$ROOT,MODEL=EleutherAI/pythia-2.8b,STUDENT=EleutherAI/pythia-410m,RUN_TAG=pythia-2.8b,$common_safe cluster/qsub_run.sh")"

echo "Submitted Pythia C2/C3 student retries:"
echo "  pythia-1.4b: $py14"
echo "  pythia-2.8b: $py28"
echo
echo "Use: qstat -u \$USER"
REMOTE
