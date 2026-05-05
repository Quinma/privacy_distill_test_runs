#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
EVAL_HELPER="${EVAL_HELPER:-$ROOT/scripts/eval_company_losses.py}"
FINALIZER="${FINALIZER:-$ROOT/scripts/finalize_placebo_attack.py}"

FAMILIES="${FAMILIES:-pythia neo}"
BF16="${BF16:-1}"
MAX_LENGTH="${MAX_LENGTH:-512}"
RETAIN_HOLDOUT_DATA="${RETAIN_HOLDOUT_DATA:-$ROOT/data/datasets/eval_retain_holdout}"

PYTHIA_MODEL="${PYTHIA_MODEL:-EleutherAI/pythia-1.4b}"
PYTHIA_STUDENT="${PYTHIA_STUDENT:-EleutherAI/pythia-410m}"
PYTHIA_RUN_TAG="${PYTHIA_RUN_TAG:-pythia-1.4b}"
PYTHIA_DATASETS_DIR="${PYTHIA_DATASETS_DIR:-$ROOT/data/datasets/$PYTHIA_RUN_TAG}"
PYTHIA_OUT_ROOT="${PYTHIA_OUT_ROOT:-$ROOT/outputs/$PYTHIA_RUN_TAG}"
PYTHIA_SEEDS="${PYTHIA_SEEDS:-13 17 19}"
PYTHIA_VISIBLE_GPUS="${PYTHIA_VISIBLE_GPUS:-0,1,2}"
PYTHIA_UNLEARN_NPROC="${PYTHIA_UNLEARN_NPROC:-3}"
PYTHIA_DISTILL_NPROC="${PYTHIA_DISTILL_NPROC:-3}"
PYTHIA_EVAL_GPU="${PYTHIA_EVAL_GPU:-3}"
PYTHIA_UNLEARN_OPTIM="${PYTHIA_UNLEARN_OPTIM:-adamw}"
PYTHIA_DISTILL_OPTIM="${PYTHIA_DISTILL_OPTIM:-adamw_torch}"

NEO_MODEL="${NEO_MODEL:-EleutherAI/gpt-neo-1.3B}"
NEO_STUDENT="${NEO_STUDENT:-EleutherAI/gpt-neo-125M}"
NEO_RUN_TAG="${NEO_RUN_TAG:-gpt-neo-1.3b-local}"
NEO_DATASETS_DIR="${NEO_DATASETS_DIR:-$ROOT/data/datasets/gpt-neo-fixed-20260419}"
NEO_OUT_ROOT="${NEO_OUT_ROOT:-$ROOT/outputs/$NEO_RUN_TAG}"
NEO_SEEDS="${NEO_SEEDS:-17 19}"
NEO_VISIBLE_GPUS="${NEO_VISIBLE_GPUS:-0,1,2}"
NEO_UNLEARN_NPROC="${NEO_UNLEARN_NPROC:-3}"
NEO_DISTILL_NPROC="${NEO_DISTILL_NPROC:-3}"
NEO_EVAL_GPU="${NEO_EVAL_GPU:-3}"
NEO_UNLEARN_OPTIM="${NEO_UNLEARN_OPTIM:-adamw}"
NEO_DISTILL_OPTIM="${NEO_DISTILL_OPTIM:-adamw_torch}"

UNLEARN_LR="${UNLEARN_LR:-2e-5}"
UNLEARN_EPOCHS="${UNLEARN_EPOCHS:-3}"
RETAIN_WEIGHT="${RETAIN_WEIGHT:-1.0}"
NPO_BETA="${NPO_BETA:-0.1}"
PER_DEVICE_BATCH="${PER_DEVICE_BATCH:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
DISTILL_EPOCHS="${DISTILL_EPOCHS:-3}"
DISTILL_LR="${DISTILL_LR:-2e-5}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
DISTILL_TEACHER_DEVICE="${DISTILL_TEACHER_DEVICE:-cuda}"
DISTILL_TEACHER_DTYPE="${DISTILL_TEACHER_DTYPE:-float32}"
DISTILL_MAX_GRAD_NORM="${DISTILL_MAX_GRAD_NORM:-1.0}"

NUM_FORGET="${NUM_FORGET:-50}"
MIN_TOKENS="${MIN_TOKENS:-200000}"
REUSE_EXISTING_FORGET="${REUSE_EXISTING_FORGET:-1}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/logs/seed_placebo_c1_retain_$(date +%Y%m%d_%H%M%S)}"
RUN_LOG="$LOG_DIR/run.log"

mkdir -p "$LOG_DIR"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$RUN_LOG"; }

bf16_flag() {
  if [[ "$BF16" == "1" ]]; then
    printf '%s' "--bf16"
  fi
}

python_has_module() {
  local module_name="$1"
  "$PY" - "$module_name" <<'PY'
import importlib.util, sys
name = sys.argv[1]
print("1" if importlib.util.find_spec(name) is not None else "0")
PY
}

resolve_distill_optim() {
  local requested="$1"
  if [[ "$requested" == "adamw_8bit" && "$(python_has_module bitsandbytes)" != "1" ]]; then
    log "WARN: bitsandbytes not available in $PY; falling back DISTILL_OPTIM from adamw_8bit to adamw_torch"
    printf "%s\n" "adamw_torch"
    return 0
  fi
  printf "%s\n" "$requested"
}

model_ready() {
  local dir="$1"
  [[ -f "$dir/model.safetensors" || -f "$dir/pytorch_model.bin" || -f "$dir/model.safetensors.index.json" || -f "$dir/model-00001-of-00002.safetensors" || -f "$dir/model-00001-of-00003.safetensors" ]]
}

require_dir() {
  local path="$1"
  if [[ ! -d "$path" ]]; then
    echo "ERROR: missing directory: $path" >&2
    exit 1
  fi
}

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing file: $path" >&2
    exit 1
  fi
}

run_logged() {
  local name="$1"
  local log_file="$LOG_DIR/${name}.log"
  shift
  log "START $name"
  echo "[$(ts)] CMD: $*" | tee -a "$log_file"
  set +e
  bash -lc "$*" 2>&1 | tee -a "$log_file"
  local status=${PIPESTATUS[0]}
  set -e
  if [[ $status -ne 0 ]]; then
    log "FAIL $name (exit $status)"
    return $status
  fi
  log "DONE $name"
}

ensure_reference_json() {
  local ref_model="$1"
  local source_json="$2"
  local output_json="$3"
  local eval_gpu="$4"
  local dataset_path="$5"

  if [[ -f "$output_json" ]]; then
    return 0
  fi
  if [[ -f "$source_json" ]]; then
    cp "$source_json" "$output_json"
    return 0
  fi
  require_dir "$ref_model"
  run_logged "$(basename "$output_json" .json)" \
    "CUDA_VISIBLE_DEVICES=$eval_gpu $PY '$EVAL_HELPER' --model '$ref_model' --dataset '$dataset_path' --output '$output_json' --batch-size 8 --max-length $MAX_LENGTH --max-chars 32768 $(bf16_flag)"
}

build_seed_summary() {
  local family="$1"
  local out_root="$2"
  local seeds="$3"
  local summary_path="$out_root/seed_c6_placebo_deletion_attack_target_vs_retain_c1ref_summary.json"

  "$PY" - "$family" "$out_root" "$summary_path" $seeds <<'PY'
import json
import statistics
import sys
from pathlib import Path

family = sys.argv[1]
out_root = Path(sys.argv[2])
summary_path = Path(sys.argv[3])
seeds = [int(x) for x in sys.argv[4:]]

rows = []
for seed in seeds:
    path = out_root / "seed_reps" / f"seed_{seed}" / "placebo_c6_c1_retain" / "mia_c6_placebo_deletion_attack_target_vs_retain_c1ref.json"
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    rows.append({
        "seed": seed,
        "auroc": float(data["auroc"]),
        "ci_low": float(data["bootstrap_ci_95"][0]),
        "ci_high": float(data["bootstrap_ci_95"][1]),
        "target_mean_delta": float(data["target_mean_delta"]),
        "retain_mean_delta": float(data["retain_mean_delta"]),
        "file": str(path),
    })

if not rows:
    raise SystemExit("no seeded placebo C1-reference results found")

aurocs = [row["auroc"] for row in rows]
result = {
    "family": family,
    "attack": "placebo_student_deletion_target_vs_retain_c1ref",
    "reference": "C1",
    "control": "disjoint_wrong_target_placebo_npo",
    "negative_class": "retained_companies",
    "positive_class": "deleted_targets",
    "seeds": rows,
    "mean_auroc": float(statistics.mean(aurocs)),
    "sd_auroc": float(statistics.pstdev(aurocs)) if len(aurocs) > 1 else 0.0,
    "n_runs": len(rows),
}
summary_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
PY
}

run_seed_family() {
  local family="$1"
  local model="$2"
  local student="$3"
  local datasets_dir="$4"
  local out_root="$5"
  local seeds="$6"
  local visible_gpus="$7"
  local unlearn_nproc="$8"
  local distill_nproc="$9"
  local eval_gpu="${10}"
  local unlearn_optim="${11}"
  local distill_optim="${12}"
  distill_optim="$(resolve_distill_optim "$distill_optim")"

  require_dir "$datasets_dir/distill"
  require_dir "$datasets_dir/eval_target_holdout"
  require_dir "$datasets_dir/teacher_c2"
  require_dir "$RETAIN_HOLDOUT_DATA"
  require_file "$datasets_dir/splits.json"
  require_file "$datasets_dir/holdout_map.json"

  for seed in $seeds; do
    local seed_root="$out_root/seed_reps/seed_${seed}"
    local seed_tag="${family}_seed${seed}"
    local c1_teacher="$seed_root/teachers/c1"
    local c1_student="$seed_root/students/c1"
    local c1_target_seed_json="$seed_root/mia/c1_student.json"
    local c1_retain_seed_json="$seed_root/mia_retain/c1_student_retain.json"

    local placebo_root="$seed_root/placebo_c6_c1_retain"
    local forget_out="$placebo_root/random_forget_train"
    local forget_ciks="$placebo_root/placebo_forget_ciks.json"
    local existing_forget_out="$seed_root/c5r_a03/random_forget_train"
    local existing_forget_ciks="$seed_root/c5r_a03/c5r_forget_ciks.json"
    local base_forget_out="$datasets_dir/random_forget_train"
    local base_forget_ciks="$datasets_dir/c5r_forget_ciks.json"
    local unlearn_out="$placebo_root/teachers/c6_placebo_unlearn"
    local student_out="$placebo_root/students/c6_placebo"
    local mia_dir="$placebo_root/mia"
    local retain_dir="$placebo_root/mia_retain"
    local ref_dir="$placebo_root/reference_eval"
    local placebo_target_json="$mia_dir/c6_placebo_student_target.json"
    local placebo_retain_json="$retain_dir/c6_placebo_student_retain.json"
    local ref_target_json="$ref_dir/c1_student_target.json"
    local ref_retain_json="$ref_dir/c1_student_retain.json"
    local final_json="$placebo_root/mia_c6_placebo_deletion_attack_target_vs_retain_c1ref.json"

    mkdir -p "$placebo_root" "$mia_dir" "$retain_dir" "$ref_dir"

    require_dir "$c1_teacher"

    if [[ ! -d "$forget_out" && "$REUSE_EXISTING_FORGET" == "1" && -d "$existing_forget_out" ]]; then
      log "${seed_tag}: reusing existing random-forget dataset at $existing_forget_out"
      ln -sfn "$existing_forget_out" "$forget_out"
      if [[ -f "$existing_forget_ciks" ]]; then
        cp "$existing_forget_ciks" "$forget_ciks"
      fi
    elif [[ ! -d "$forget_out" && "$REUSE_EXISTING_FORGET" == "1" && -d "$base_forget_out" ]]; then
      log "${seed_tag}: reusing base random-forget dataset at $base_forget_out"
      ln -sfn "$base_forget_out" "$forget_out"
      if [[ -f "$base_forget_ciks" ]]; then
        cp "$base_forget_ciks" "$forget_ciks"
      fi
    elif [[ ! -d "$forget_out" ]]; then
      run_logged "${seed_tag}_build_random_forget" \
        "$PY '$ROOT/src/build_random_forget.py' --dataset 'bradfordlevy/BeanCounter' --config 'clean' --split 'train' --cik-map '$ROOT/data/sec_index_10k.jsonl' --tokenizer '$model' --form-types '10-K' --splits '$datasets_dir/splits.json' --stats-path '$ROOT/data/bean_counter_stats.jsonl' --num-forget $NUM_FORGET --min-tokens $MIN_TOKENS --max-length $MAX_LENGTH --max-tokens-per-company 0 --seed $seed --output '$forget_out' --ciks-output '$forget_ciks'"
    else
      log "SKIP ${seed_tag}_build_random_forget (dataset exists)"
    fi

    if ! model_ready "$unlearn_out"; then
      if [[ "$unlearn_nproc" -gt 1 ]]; then
        run_logged "${seed_tag}_unlearn_placebo" \
          "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$visible_gpus $PY -m torch.distributed.run --nproc_per_node=$unlearn_nproc '$ROOT/src/unlearn_teacher.py' --model '$c1_teacher' --ref-model '$c1_teacher' --forget-dataset '$forget_out' --retain-dataset '$datasets_dir/teacher_c2' --output '$unlearn_out' --method npo --npo-beta $NPO_BETA --beta $RETAIN_WEIGHT --epochs $UNLEARN_EPOCHS --lr $UNLEARN_LR --optim '$unlearn_optim' --batch-size $PER_DEVICE_BATCH --grad-accum $GRAD_ACCUM --seed $seed --fsdp $(bf16_flag)"
      else
        run_logged "${seed_tag}_unlearn_placebo" \
          "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$eval_gpu $PY '$ROOT/src/unlearn_teacher.py' --model '$c1_teacher' --ref-model '$c1_teacher' --forget-dataset '$forget_out' --retain-dataset '$datasets_dir/teacher_c2' --output '$unlearn_out' --method npo --npo-beta $NPO_BETA --beta $RETAIN_WEIGHT --epochs $UNLEARN_EPOCHS --lr $UNLEARN_LR --optim '$unlearn_optim' --batch-size $PER_DEVICE_BATCH --grad-accum $GRAD_ACCUM --seed $seed $(bf16_flag)"
      fi
    else
      log "SKIP ${seed_tag}_unlearn_placebo (model exists)"
    fi

    if ! model_ready "$student_out"; then
      if [[ "$distill_nproc" -gt 1 ]]; then
        run_logged "${seed_tag}_distill_placebo" \
          "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$visible_gpus $PY -m torch.distributed.run --nproc_per_node=$distill_nproc '$ROOT/src/distill_student.py' --teacher '$unlearn_out' --student '$student' --dataset '$datasets_dir/distill' --output '$student_out' --teacher-device '$DISTILL_TEACHER_DEVICE' --teacher-dtype '$DISTILL_TEACHER_DTYPE' --max-length $MAX_LENGTH --epochs $DISTILL_EPOCHS --lr $DISTILL_LR --warmup-steps $WARMUP_STEPS --per-device-batch $PER_DEVICE_BATCH --grad-accum $GRAD_ACCUM --optim '$distill_optim' --seed $seed --save-steps 999999 --logging-steps 50 --max-grad-norm $DISTILL_MAX_GRAD_NORM $(bf16_flag)"
      else
        run_logged "${seed_tag}_distill_placebo" \
          "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=$eval_gpu $PY '$ROOT/src/distill_student.py' --teacher '$unlearn_out' --student '$student' --dataset '$datasets_dir/distill' --output '$student_out' --teacher-device '$DISTILL_TEACHER_DEVICE' --teacher-dtype '$DISTILL_TEACHER_DTYPE' --max-length $MAX_LENGTH --epochs $DISTILL_EPOCHS --lr $DISTILL_LR --warmup-steps $WARMUP_STEPS --per-device-batch $PER_DEVICE_BATCH --grad-accum $GRAD_ACCUM --optim '$distill_optim' --seed $seed --save-steps 999999 --logging-steps 50 --max-grad-norm $DISTILL_MAX_GRAD_NORM $(bf16_flag)"
      fi
    else
      log "SKIP ${seed_tag}_distill_placebo (model exists)"
    fi

    ensure_reference_json "$c1_student" "$c1_target_seed_json" "$ref_target_json" "$eval_gpu" "$datasets_dir/eval_target_holdout"
    ensure_reference_json "$c1_student" "$c1_retain_seed_json" "$ref_retain_json" "$eval_gpu" "$RETAIN_HOLDOUT_DATA"

    if [[ ! -f "$placebo_target_json" ]]; then
      run_logged "${seed_tag}_eval_placebo_target" \
        "CUDA_VISIBLE_DEVICES=$eval_gpu $PY '$EVAL_HELPER' --model '$student_out' --dataset '$datasets_dir/eval_target_holdout' --output '$placebo_target_json' --batch-size 8 --max-length $MAX_LENGTH --max-chars 32768 $(bf16_flag)"
    else
      log "SKIP ${seed_tag}_eval_placebo_target (output exists)"
    fi

    if [[ ! -f "$placebo_retain_json" ]]; then
      run_logged "${seed_tag}_eval_placebo_retain" \
        "CUDA_VISIBLE_DEVICES=$eval_gpu $PY '$EVAL_HELPER' --model '$student_out' --dataset '$RETAIN_HOLDOUT_DATA' --output '$placebo_retain_json' --batch-size 8 --max-length $MAX_LENGTH --max-chars 32768 $(bf16_flag)"
    else
      log "SKIP ${seed_tag}_eval_placebo_retain (output exists)"
    fi

    if [[ ! -f "$final_json" ]]; then
      run_logged "${seed_tag}_attack_c1ref" \
        "$PY '$FINALIZER' --placebo-target '$placebo_target_json' --ref-target '$ref_target_json' --placebo-retain '$placebo_retain_json' --ref-retain '$ref_retain_json' --reference C1 --seed $seed --output '$final_json'"
    else
      log "SKIP ${seed_tag}_attack_c1ref (output exists)"
    fi
  done

  build_seed_summary "$family" "$out_root" "$seeds"
}

log "Seeded placebo C1-reference retain-pool run start: families=$FAMILIES"

run_family_pythia() {
  run_seed_family \
    "pythia-1.4B" \
    "$PYTHIA_MODEL" \
    "$PYTHIA_STUDENT" \
    "$PYTHIA_DATASETS_DIR" \
    "$PYTHIA_OUT_ROOT" \
    "$PYTHIA_SEEDS" \
    "$PYTHIA_VISIBLE_GPUS" \
    "$PYTHIA_UNLEARN_NPROC" \
    "$PYTHIA_DISTILL_NPROC" \
    "$PYTHIA_EVAL_GPU" \
    "$PYTHIA_UNLEARN_OPTIM" \
    "$PYTHIA_DISTILL_OPTIM"
}

run_family_neo() {
  run_seed_family \
    "neo-1.3B" \
    "$NEO_MODEL" \
    "$NEO_STUDENT" \
    "$NEO_DATASETS_DIR" \
    "$NEO_OUT_ROOT" \
    "$NEO_SEEDS" \
    "$NEO_VISIBLE_GPUS" \
    "$NEO_UNLEARN_NPROC" \
    "$NEO_DISTILL_NPROC" \
    "$NEO_EVAL_GPU" \
    "$NEO_UNLEARN_OPTIM" \
    "$NEO_DISTILL_OPTIM"
}

for family in $FAMILIES; do
  case "$family" in
    pythia) run_family_pythia ;;
    neo) run_family_neo ;;
    *)
      echo "ERROR: unknown family '$family' (expected pythia and/or neo)" >&2
      exit 1
      ;;
  esac
done

log "Seeded placebo C1-reference retain-pool run complete."
