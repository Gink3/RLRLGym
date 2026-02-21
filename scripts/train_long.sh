#!/usr/bin/env bash
set -euo pipefail

python3 -m train \
  --episodes 500 \
  --max-steps 150 \
  --seed 0 \
  --output-dir outputs/train/long \
  --networks-path data/agent_networks.json \
  "$@"
