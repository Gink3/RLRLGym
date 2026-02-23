#!/usr/bin/env bash
set -euo pipefail

PY_BIN="python3"
if [[ -x ".venv/bin/python" ]]; then
  PY_BIN=".venv/bin/python"
fi


"$PY_BIN" -m train \
  --backend rllib \
  --iterations 200 \
  --max-steps 150 \
  --seed 0 \
  --output-dir outputs/train/long \
  "$@"
