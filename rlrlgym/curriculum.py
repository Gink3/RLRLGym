"""Curriculum phase config loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


REQUIRED_PHASE_FIELDS = {
    "until_episode",
    "width",
    "height",
    "max_steps",
    "monster_density",
    "chest_density",
}


def load_curriculum_phases(path: str | Path) -> List[Dict[str, object]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Curriculum JSON must be an object")
    if "schema_version" not in raw or not isinstance(raw["schema_version"], int):
        raise ValueError("Curriculum JSON requires integer schema_version")
    if "phases" not in raw or not isinstance(raw["phases"], list):
        raise ValueError("Curriculum JSON requires array 'phases'")

    out: List[Dict[str, object]] = []
    for idx, row in enumerate(raw["phases"]):
        if not isinstance(row, dict):
            raise ValueError(f"phases[{idx}] must be an object")
        missing = REQUIRED_PHASE_FIELDS - set(row.keys())
        if missing:
            miss = ", ".join(sorted(missing))
            raise ValueError(f"phases[{idx}] missing required field(s): {miss}")
        out.append(
            {
                "name": str(row.get("name", f"phase_{idx + 1}")),
                "until_episode": int(row["until_episode"]),
                "width": int(row["width"]),
                "height": int(row["height"]),
                "max_steps": int(row["max_steps"]),
                "monster_density": float(row["monster_density"]),
                "chest_density": float(row["chest_density"]),
            }
        )
        if "combat_training_mode" in row:
            out[-1]["combat_training_mode"] = bool(row["combat_training_mode"])
        if "hunger_tick_enabled" in row:
            out[-1]["hunger_tick_enabled"] = bool(row["hunger_tick_enabled"])
        if "missed_attack_opportunity_penalty" in row:
            out[-1]["missed_attack_opportunity_penalty"] = float(
                row["missed_attack_opportunity_penalty"]
            )

    if not out:
        raise ValueError("At least one curriculum phase is required")
    return out
