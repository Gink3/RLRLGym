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
  --iterations 4000 \
  --width 50 \
  --height 50 \
  --max-steps 300 \
  --no-curriculum \
  --seed 1 \
  --output-dir outputs/train/overnight \
  "$@"
