#!/usr/bin/env bash
set -euo pipefail

export BASE_MODEL=${BASE_MODEL:-EleutherAI/pythia-2.8b}
export STUDENT_MODEL=${STUDENT_MODEL:-EleutherAI/pythia-410m}
export TOKENIZER=${TOKENIZER:-$BASE_MODEL}
export RUN_TAG=${RUN_TAG:-pythia-2.8b}
export MAX_TOKENS_PER_COMPANY=${MAX_TOKENS_PER_COMPANY:-500000}
export GRAD_CHECKPOINTING=${GRAD_CHECKPOINTING:-1}
export BATCH=${BATCH:-1}
export ACCUM=${ACCUM:-32}
export FSDP=${FSDP:-1}

bash scripts/run_pipeline.sh
