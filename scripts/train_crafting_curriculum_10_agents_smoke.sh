#!/usr/bin/env bash
set -euo pipefail

PY_BIN="python3"
if [[ -x ".venv/bin/python" ]]; then
  PY_BIN=".venv/bin/python"
fi

SCENARIO_DIR="data/scenarios/crafting_curriculum_10_agents"
TMP_CURRICULUM="$(mktemp "${TMPDIR:-/tmp}/crafting_curriculum_smoke.XXXXXX.json")"
cleanup() {
  rm -f "${TMP_CURRICULUM}"
}
trap cleanup EXIT

SCENARIO_DIR_ABS="$(cd "${SCENARIO_DIR}" && pwd)"
"$PY_BIN" - <<'PY' "${SCENARIO_DIR_ABS}" "${TMP_CURRICULUM}"
import json
import sys
from pathlib import Path

scenario_dir = Path(sys.argv[1]).resolve()
out_path = Path(sys.argv[2]).resolve()
source = scenario_dir / "curriculum_with_static_maps.json"
raw = json.loads(source.read_text(encoding="utf-8"))
phases = raw.get("phases", [])
if not isinstance(phases, list) or not phases:
    raise SystemExit(f"Invalid curriculum source: {source}")

smoke_phases = []
for idx, phase in enumerate(phases, start=1):
    if not isinstance(phase, dict):
        continue
    row = dict(phase)
    row["name"] = f"{row.get('name', f'phase_{idx}')}_smoke"
    row["until_episode"] = 0 if idx == len(phases) else idx
    row["width"] = min(int(row.get("width", 32)), 96)
    row["height"] = min(int(row.get("height", 32)), 96)
    row["max_steps"] = min(int(row.get("max_steps", 80)), 40)
    row.pop("static_map_path", None)
    smoke_phases.append(row)

payload = {"schema_version": 1, "phases": smoke_phases}
out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

echo "Smoke curriculum: ${TMP_CURRICULUM}"
echo "Smoke curriculum runs all 7 phases with one episode per phase."
echo "Replay capture is disabled in this smoke profile path to avoid serialization skew."

"$PY_BIN" -m train \
  --backend rllib \
  --algo ppo_masked \
  --iterations 7 \
  --scenario-path "${SCENARIO_DIR}" \
  --curriculum-path "${TMP_CURRICULUM}" \
  --shared-policy \
  --num-rollout-workers 0 \
  --train-batch-size 400 \
  --sgd-minibatch-size 128 \
  --num-sgd-iter 1 \
  --rollout-fragment-length 50 \
  --sample-timeout-s 300 \
  --replay-save-every 0 \
  --no-save-latest-replay \
  --seed 0 \
  --no-aim \
  --output-dir outputs/train/crafting_curriculum_10_agents_smoke \
  "$@"
