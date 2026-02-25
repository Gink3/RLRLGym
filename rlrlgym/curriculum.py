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

OPTIONAL_PHASE_FIELDS = {
    "combat_training_mode": bool,
    "hunger_tick_enabled": bool,
    "missed_attack_opportunity_penalty": float,
    "new_tile_seen_reward": float,
    "frontier_step_reward": float,
    "stagnation_penalty": float,
    "stagnation_threshold_steps": int,
    "repeat_visit_penalty": float,
    "repeat_visit_window": int,
    "move_bias_reward": float,
    "wait_no_enemy_penalty": float,
    "wait_safe_hunger_ratio": float,
    "first_enemy_seen_bonus": float,
    "enemy_visible_reward": float,
    "enemy_distance_delta_reward_scale": float,
    "enemy_distance_delta_clip": float,
    "lost_enemy_penalty": float,
    "timeout_tie_penalty": float,
    "engagement_bonus": float,
    "monster_sight_range": int,
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
        for key, caster in OPTIONAL_PHASE_FIELDS.items():
            if key in row:
                out[-1][key] = caster(row[key])

    if not out:
        raise ValueError("At least one curriculum phase is required")
    return out
