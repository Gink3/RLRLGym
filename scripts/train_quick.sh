#!/usr/bin/env bash
set -euo pipefail

python3 -m train \
  --episodes 20 \
  --max-steps 80 \
  --seed 0 \
  --output-dir outputs/train/quick \
  --networks-path data/agent_networks.json \
  "$@"
