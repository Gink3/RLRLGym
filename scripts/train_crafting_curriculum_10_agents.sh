#!/usr/bin/env bash
set -euo pipefail

PY_BIN="python3"
if [[ -x ".venv/bin/python" ]]; then
  PY_BIN=".venv/bin/python"
fi

"$PY_BIN" -m train \
  --backend rllib \
  --algo ppo_masked \
  --iterations 200 \
  --scenario-path data/scenarios/crafting_curriculum_10_agents \
  --curriculum-path data/base/curriculum_phases_crafting_1000.json \
  --shared-policy \
  --num-rollout-workers 2 \
  --train-batch-size 8000 \
  --sgd-minibatch-size 1024 \
  --num-sgd-iter 10 \
  --replay-save-every 5000 \
  --seed 0 \
  --output-dir outputs/train/crafting_curriculum_10_agents \
  "$@"
