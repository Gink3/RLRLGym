#!/usr/bin/env bash
set -euo pipefail

# Full training run intended for stronger convergence than quick/default presets.
# Usage:
#   ./scripts/train_full.sh
#   ./scripts/train_full.sh --seed 3 --output-dir outputs/train/full_seed3

python3 -m train \
  --episodes 2000 \
  --max-steps 200 \
  --seed 0 \
  --output-dir outputs/train/full \
  --networks-path data/agent_networks.json \
  "$@"
