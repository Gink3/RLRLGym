#!/usr/bin/env bash
set -euo pipefail

PY_BIN="python3"
if [[ -x ".venv/bin/python" ]]; then
  PY_BIN=".venv/bin/python"
fi


# Full training run intended for stronger convergence than quick/default presets.
# Usage:
#   ./scripts/train_full.sh
#   ./scripts/train_full.sh --seed 3 --output-dir outputs/train/full_seed3

"$PY_BIN" -m train \
  --backend rllib \
  --iterations 1000 \
  --max-steps 200 \
  --seed 0 \
  --output-dir outputs/train/full \
  "$@"
