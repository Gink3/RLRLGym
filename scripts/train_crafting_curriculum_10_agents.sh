#!/usr/bin/env bash
set -euo pipefail

PY_BIN="python3"
if [[ -x ".venv/bin/python" ]]; then
  PY_BIN=".venv/bin/python"
fi

SCENARIO_DIR="data/scenarios/crafting_curriculum_10_agents"
CURRICULUM_PATH="data/base/curriculum_phases_crafting_1000.json"
if [[ -f "${SCENARIO_DIR}/curriculum_with_static_maps.json" ]]; then
  CURRICULUM_PATH="${SCENARIO_DIR}/curriculum_with_static_maps.json"
fi

"$PY_BIN" -m train \
  --backend rllib \
  --algo ppo_masked \
  --iterations 200 \
  --scenario-path "${SCENARIO_DIR}" \
  --curriculum-path "${CURRICULUM_PATH}" \
  --shared-policy \
  --num-rollout-workers 2 \
  --train-batch-size 8000 \
  --sgd-minibatch-size 1024 \
  --num-sgd-iter 10 \
  --rollout-fragment-length 100 \
  --sample-timeout-s 600 \
  --replay-save-every 5000 \
  --seed 0 \
  --output-dir outputs/train/crafting_curriculum_10_agents \
  "$@"
