#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-outputs/train/quick}"
PORT="${2:-8000}"

if [[ ! -d "$RUN_DIR" ]]; then
  echo "Run directory not found: $RUN_DIR" >&2
  echo "Usage: $0 [run_dir] [port]" >&2
  exit 1
fi

if [[ ! -f "$RUN_DIR/dashboard.html" ]]; then
  echo "dashboard.html not found in: $RUN_DIR" >&2
  echo "Run training first, then retry." >&2
  exit 1
fi

ABS_DIR="$(cd "$RUN_DIR" && pwd)"

echo "Serving dashboard from: $ABS_DIR"
echo "Open: http://127.0.0.1:${PORT}/dashboard.html"

cd "$ABS_DIR"
python3 -m http.server "$PORT"
