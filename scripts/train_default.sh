#!/usr/bin/env bash
set -euo pipefail

python3 -m train \
  --episodes 100 \
  --max-steps 120 \
  --seed 0 \
  --output-dir outputs/train/default \
  --networks-path data/agent_networks.json \
  "$@"
