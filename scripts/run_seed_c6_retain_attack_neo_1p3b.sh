#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-$ROOT/.venv/bin/python}"
SEED_RUNNER="${SEED_RUNNER:-$ROOT/scripts/run_seed_c6_npo_neo_1p3b.sh}"
EVAL_HELPER="${EVAL_HELPER:-$ROOT/scripts/eval_company_losses.py}"
FINALIZER="${FINALIZER:-$ROOT/scripts/finalize_deletion_attack.py}"

RUN_TAG="${RUN_TAG:-gpt-neo-1.3b-local}"
DATASETS_DIR="${DATASETS_DIR:-$ROOT/data/datasets/gpt-neo-fixed-20260419}"
SEED_ROOT="${SEED_ROOT:-$ROOT/outputs/$RUN_TAG/seed_reps}"
SEEDS="${SEEDS:-17 19}"
RETAIN_HOLDOUT_DATA="${RETAIN_HOLDOUT_DATA:-$ROOT/data/datasets/eval_retain_holdout}"
MAX_LENGTH="${MAX_LENGTH:-512}"
BF16="${BF16:-1}"
EVAL_GPU="${EVAL_GPU:-3}"
LOG_DIR="${LOG_DIR:-$ROOT/outputs/logs/${RUN_TAG}_seed_c6_retain_$(date +%Y%m%d_%H%M%S)}"
RUN_LOG="$LOG_DIR/run.log"
mkdir -p "$LOG_DIR"

ts() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(ts)] $*" | tee -a "$RUN_LOG"; }

bf16_flag() {
  if [[ "$BF16" == "1" ]]; then
    printf '%s' "--bf16"
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

build_seed_summary() {
  local out_root="$1"
  local reference="$2"
  local summary_path="$3"
  shift 3
  "$PY" - "$out_root" "$reference" "$summary_path" "$@" <<'PY'
import json
import statistics
import sys
from pathlib import Path

out_root = Path(sys.argv[1])
reference = sys.argv[2]
summary_path = Path(sys.argv[3])
seeds = [int(x) for x in sys.argv[4:]]
suffix = "_c1ref" if reference == "C1" else ""
rows = []
for seed in seeds:
    path = out_root / f"seed_{seed}" / f"mia_c6_deletion_attack_target_vs_retain{suffix}.json"
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
    raise SystemExit("no seeded canonical retain-pool results found")
aurocs = [row["auroc"] for row in rows]
result = {
    "family": "neo-1.3B",
    "attack": f"canonical_student_deletion_target_vs_retain_{reference.lower()}ref",
    "reference": reference,
    "control": "canonical_c6_npo",
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

log "Neo canonical seeded retain-pool start: seeds=$SEEDS eval_gpu=$EVAL_GPU"

run_logged "seeded_c6_core" \
  "SEEDS='$SEEDS' DATASETS_DIR='$DATASETS_DIR' SEED_ROOT='$SEED_ROOT' EVAL_GPU='$EVAL_GPU' LOG_DIR='$LOG_DIR/core' bash '$SEED_RUNNER'"

for seed in $SEEDS; do
  seed_root="$SEED_ROOT/seed_${seed}"
  mia_dir="$seed_root/mia"
  retain_dir="$seed_root/mia_retain"
  mkdir -p "$retain_dir"

  c1_model="$seed_root/students/c1"
  c3_model="$seed_root/students/c3"
  c6_model="$seed_root/students/c6"
  c1_target="$mia_dir/c1_student.json"
  c3_target="$mia_dir/c3_student.json"
  c6_target="$mia_dir/c6_student.json"
  c1_retain="$retain_dir/c1_student_retain.json"
  c3_retain="$retain_dir/c3_student_retain.json"
  c6_retain="$retain_dir/c6_student_retain.json"
  c3_final="$seed_root/mia_c6_deletion_attack_target_vs_retain.json"
  c1_final="$seed_root/mia_c6_deletion_attack_target_vs_retain_c1ref.json"

  if [[ ! -f "$c1_retain" ]]; then
    run_logged "seed_${seed}_retain_c1" \
      "CUDA_VISIBLE_DEVICES=$EVAL_GPU '$PY' '$EVAL_HELPER' --model '$c1_model' --dataset '$RETAIN_HOLDOUT_DATA' --output '$c1_retain' --batch-size 8 --max-length $MAX_LENGTH --max-chars 32768 $(bf16_flag)"
  else
    log "SKIP seed_${seed}_retain_c1 (output exists)"
  fi

  if [[ ! -f "$c3_retain" ]]; then
    run_logged "seed_${seed}_retain_c3" \
      "CUDA_VISIBLE_DEVICES=$EVAL_GPU '$PY' '$EVAL_HELPER' --model '$c3_model' --dataset '$RETAIN_HOLDOUT_DATA' --output '$c3_retain' --batch-size 8 --max-length $MAX_LENGTH --max-chars 32768 $(bf16_flag)"
  else
    log "SKIP seed_${seed}_retain_c3 (output exists)"
  fi

  if [[ ! -f "$c6_retain" ]]; then
    run_logged "seed_${seed}_retain_c6" \
      "CUDA_VISIBLE_DEVICES=$EVAL_GPU '$PY' '$EVAL_HELPER' --model '$c6_model' --dataset '$RETAIN_HOLDOUT_DATA' --output '$c6_retain' --batch-size 8 --max-length $MAX_LENGTH --max-chars 32768 $(bf16_flag)"
  else
    log "SKIP seed_${seed}_retain_c6 (output exists)"
  fi

  if [[ ! -f "$c3_final" ]]; then
    run_logged "seed_${seed}_attack_c3ref" \
      "'$PY' '$FINALIZER' --c6-target '$c6_target' --ref-target '$c3_target' --c6-retain '$c6_retain' --ref-retain '$c3_retain' --reference C3 --seed $seed --control canonical_c6_npo --output '$c3_final'"
  else
    log "SKIP seed_${seed}_attack_c3ref (output exists)"
  fi

  if [[ ! -f "$c1_final" ]]; then
    run_logged "seed_${seed}_attack_c1ref" \
      "'$PY' '$FINALIZER' --c6-target '$c6_target' --ref-target '$c1_target' --c6-retain '$c6_retain' --ref-retain '$c1_retain' --reference C1 --seed $seed --control canonical_c6_npo --output '$c1_final'"
  else
    log "SKIP seed_${seed}_attack_c1ref (output exists)"
  fi
done

build_seed_summary "$SEED_ROOT" C3 "$ROOT/outputs/$RUN_TAG/seed_c6_deletion_attack_target_vs_retain_summary.json" $SEEDS
build_seed_summary "$SEED_ROOT" C1 "$ROOT/outputs/$RUN_TAG/seed_c6_deletion_attack_target_vs_retain_c1ref_summary.json" $SEEDS

log "Neo canonical seeded retain-pool complete."
