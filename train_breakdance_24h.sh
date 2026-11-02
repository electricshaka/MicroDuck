#!/usr/bin/env bash
# Resume a Got a Talent training from an archived checkpoint, the way the guide's
# "Resume training" step does it (src/microduck_challenges/got_a_talent/AGENTS.md).
#
#   ./train_breakdance_24h.sh <run-name> [load-run] [checkpoint] [iterations] [num-envs]
#
# Defaults resume the sanctioned base, logs/rsl_rl/got_a_talent/hs12b_base/model_2000.pt
# (stage it byte-for-byte first: the base rule allows winners <= 2250 only).
# MICRODUCK_WARM_START keeps every curriculum at the stage the checkpoint reached;
# without it each curriculum jumps to its last stage on resume.
set -euo pipefail
cd "$(dirname "$0")"

RUN_NAME="${1:?usage: $0 <run-name> [load-run] [checkpoint] [iterations] [num-envs]}"
LOAD_RUN="${2:-hs12b_base}"
CHECKPOINT="${3:-model_2000.pt}"
ITERATIONS="${4:-3000}"
NUM_ENVS="${5:-4096}"

export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
export MICRODUCK_WARM_START=1
unset PYTORCH_NO_CUDA_MEMORY_CACHING PYTORCH_CUDA_ALLOC_CONF || true
export PYTORCH_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p logs

exec uv run --no-sync train Mjlab-GotATalent-MicroDuck \
    --env.scene.num-envs "$NUM_ENVS" \
    --agent.max-iterations "$ITERATIONS" \
    --agent.seed 1 \
    --agent.logger tensorboard \
    --agent.resume True \
    --agent.load-run "$LOAD_RUN" \
    --agent.load-checkpoint "$CHECKPOINT" \
    --agent.run-name "$RUN_NAME" 2>&1 | tee -a "logs/${RUN_NAME}.log"
