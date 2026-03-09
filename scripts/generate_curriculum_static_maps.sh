#!/usr/bin/env bash
set -euo pipefail

PY_BIN="python3"
if [[ -x ".venv/bin/python" ]]; then
  PY_BIN=".venv/bin/python"
fi

CURRICULUM_PATH="${1:-data/base/curriculum_phases_crafting_1000.json}"
SCENARIO_PATH="${2:-data/scenarios/crafting_curriculum_10_agents}"
OUTPUT_DIR="${3:-${SCENARIO_PATH%/}/maps}"
BASE_SEED="${4:-7000}"
CURRICULUM_WITH_MAPS_PATH="${5:-${SCENARIO_PATH%/}/curriculum_with_static_maps.json}"

"$PY_BIN" - "$CURRICULUM_PATH" "$SCENARIO_PATH" "$OUTPUT_DIR" "$BASE_SEED" "$CURRICULUM_WITH_MAPS_PATH" << 'PY'
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from rlrlgym.world.env import EnvConfig, PettingZooParallelRLRLGym
from rlrlgym.systems.scenario import apply_scenario_to_env_config, load_scenario


def _safe_name(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value).strip())


def _map_payload(*, name: str, seed: int, grid: list[list[str]], biomes: dict[tuple[int, int], str]) -> dict[str, object]:
    biome_rows = [
        {"position": [int(r), int(c)], "biome": str(biome)}
        for (r, c), biome in sorted(biomes.items())
        if str(biome).strip()
    ]
    return {
        "schema_version": 1,
        "map": {
            "name": str(name),
            "seed": int(seed),
            "width": int(len(grid[0]) if grid else 0),
            "height": int(len(grid)),
            "grid": [[str(tile) for tile in row] for row in grid],
            "biomes": biome_rows,
        },
    }


def main() -> None:
    if len(sys.argv) < 6:
        raise SystemExit(
            "Usage: <curriculum_path> <scenario_path> <output_dir> <base_seed> <curriculum_with_maps_path>"
        )
    curriculum_path = Path(sys.argv[1]).resolve()
    scenario_path = Path(sys.argv[2]).resolve()
    output_dir = Path(sys.argv[3]).resolve()
    base_seed = int(sys.argv[4])
    curriculum_with_maps_path = Path(sys.argv[5]).resolve()

    raw = json.loads(curriculum_path.read_text(encoding="utf-8"))
    phases = raw.get("phases", [])
    if not isinstance(phases, list) or not phases:
        raise ValueError("Curriculum must contain non-empty phases array")
    updated_curriculum = dict(raw)
    updated_curriculum["phases"] = [
        dict(x) for x in phases if isinstance(x, dict)
    ]

    base_cfg = EnvConfig.from_json(Path("data/env_config.json"))
    scenario = load_scenario(scenario_path)
    base_cfg = apply_scenario_to_env_config(base_cfg, scenario)
    scenario_dir = scenario_path if scenario_path.is_dir() else scenario_path.parent

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "schema_version": 1,
        "source_curriculum": str(curriculum_path),
        "source_scenario": str(scenario_path),
        "base_seed": int(base_seed),
        "maps": [],
    }

    phase_rows = updated_curriculum.get("phases", [])
    for idx, phase_any in enumerate(phase_rows):
        if not isinstance(phase_any, dict):
            continue
        phase = dict(phase_any)
        phase_name = str(phase.get("name", f"phase_{idx+1}")).strip() or f"phase_{idx+1}"
        seed = base_seed + idx
        cfg = copy.deepcopy(base_cfg)
        if "width" in phase:
            cfg.width = int(phase["width"])
        if "height" in phase:
            cfg.height = int(phase["height"])
        if "max_steps" in phase:
            cfg.max_steps = int(phase["max_steps"])
        cfg.render_enabled = False

        env = PettingZooParallelRLRLGym(cfg)
        env.reset(seed=seed)
        assert env.state is not None
        grid = [list(row) for row in env.state.grid]
        biomes = dict(env.state.biomes)

        file_name = f"{idx+1:02d}_{_safe_name(phase_name)}_{cfg.width}x{cfg.height}_seed{seed}.json"
        out_path = output_dir / file_name
        payload = _map_payload(
            name=f"{phase_name}_static",
            seed=seed,
            grid=grid,
            biomes=biomes,
        )
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        try:
            rel_for_phase = out_path.relative_to(scenario_dir).as_posix()
        except Exception:
            rel_for_phase = out_path.as_posix()
        phase_rows[idx]["static_map_path"] = rel_for_phase

        manifest["maps"].append(
            {
                "phase_index": int(idx + 1),
                "phase_name": phase_name,
                "seed": int(seed),
                "width": int(cfg.width),
                "height": int(cfg.height),
                "path": str(out_path),
            }
        )
        print(f"wrote {out_path}")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {manifest_path}")
    curriculum_with_maps_path.parent.mkdir(parents=True, exist_ok=True)
    curriculum_with_maps_path.write_text(
        json.dumps(updated_curriculum, indent=2), encoding="utf-8"
    )
    print(f"wrote {curriculum_with_maps_path}")


if __name__ == "__main__":
    main()
PY
