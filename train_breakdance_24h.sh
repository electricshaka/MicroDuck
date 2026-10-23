#!/usr/bin/env bash
set -euo pipefail
cd /home/cheetor/GitHub/microduck-private
export PYTHONUNBUFFERED=1
export PYTHONUTF8=1
unset PYTORCH_NO_CUDA_MEMORY_CACHING || true
unset PYTORCH_CUDA_ALLOC_CONF || true
export PYTORCH_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.8"
export OMP_NUM_THREADS=8
mkdir -p logs

exec uv run --no-sync train Mjlab-GotATalent-MicroDuck \
    --env.scene.num-envs 4096 \
    --agent.max-iterations 10000 \
    --agent.seed 1 \
    --agent.logger tensorboard \
    --agent.resume True \
    --agent.load-run 2026-10-23_05-30-24_spark_got_a_talent_breakdance_01 \
    --agent.load-checkpoint model_1250.pt \
    --agent.run-name spark_got_a_talent_breakdance_01 2>&1 | tee -a logs/breakdance_24h.log
