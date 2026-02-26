#!/usr/bin/env bash
set -euo pipefail

PY_BIN="python3"
if [[ -x ".venv/bin/python" ]]; then
  PY_BIN=".venv/bin/python"
fi


"$PY_BIN" -m train \
  --backend rllib \
  --iterations 200 \
  --max-steps 120 \
  --width 14 \
  --height 14 \
  --train-batch-size 8000 \
  --num-rollout-workers 2 \
  --shared-policy \
  --replay-save-every 5000 \
  --seed 0 \
  --output-dir outputs/train/quick \
  "$@"
