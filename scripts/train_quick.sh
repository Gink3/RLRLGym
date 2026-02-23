#!/usr/bin/env bash
set -euo pipefail

PY_BIN="python3"
if [[ -x ".venv/bin/python" ]]; then
  PY_BIN=".venv/bin/python"
fi


"$PY_BIN" -m train \
  --backend rllib \
  --iterations 10 \
  --max-steps 80 \
  --seed 0 \
  --output-dir outputs/train/quick \
  "$@"
