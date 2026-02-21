"""Agent profile definitions, reward shaping, and JSON loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


REQUIRED_PROFILE_FIELDS = {
    "name",
    "max_hp",
    "max_hunger",
    "view_radius",
    "include_grid",
    "include_stats",
    "include_inventory",
    "reward_weights",
}


@dataclass
class AgentProfile:
    name: str
    max_hp: int
    max_hunger: int
    view_radius: int
    include_grid: bool = True
    include_stats: bool = True
    include_inventory: bool = True
    reward_weights: Dict[str, float] = field(default_factory=dict)

    def reward_adjustment(self, events: List[str], died: bool) -> float:
        w = self.reward_weights
        delta = 0.0
        for event in events:
            if event == "explore":
                delta += w.get("explore", 0.0)
            elif event.startswith("loot:") or event.startswith("pickup:"):
                delta += w.get("loot", 0.0)
            elif event.startswith("interact:") or event.startswith("agent_interact:"):
                delta += w.get("interact", 0.0)
            elif event.startswith("eat:"):
                delta += w.get("eat", 0.0)
            elif event in ("wait", "wait_loop_penalty"):
                delta -= w.get("wait_penalty", 0.0)
            elif event == "stutter_penalty":
                delta -= w.get("stutter_penalty", 0.0)
        if died:
            delta -= w.get("death_penalty", 0.0)
        return delta


def load_profiles(path: str | Path) -> Dict[str, AgentProfile]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Agent profiles JSON must be an object")
    if "schema_version" not in raw or not isinstance(raw["schema_version"], int):
        raise ValueError("Agent profiles JSON requires integer schema_version")
    if "profiles" not in raw or not isinstance(raw["profiles"], list):
        raise ValueError("Agent profiles JSON requires array 'profiles'")

    profiles: Dict[str, AgentProfile] = {}
    for idx, row in enumerate(raw["profiles"]):
        if not isinstance(row, dict):
            raise ValueError(f"profile[{idx}] must be an object")
        missing = REQUIRED_PROFILE_FIELDS - set(row.keys())
        if missing:
            missing_sorted = ", ".join(sorted(missing))
            raise ValueError(f"profile[{idx}] missing required field(s): {missing_sorted}")
        if not isinstance(row["reward_weights"], dict):
            raise ValueError(f"profile[{idx}].reward_weights must be an object")

        profile = AgentProfile(
            name=str(row["name"]),
            max_hp=int(row["max_hp"]),
            max_hunger=int(row["max_hunger"]),
            view_radius=int(row["view_radius"]),
            include_grid=bool(row["include_grid"]),
            include_stats=bool(row["include_stats"]),
            include_inventory=bool(row["include_inventory"]),
            reward_weights={k: float(v) for k, v in row["reward_weights"].items()},
        )
        profiles[profile.name] = profile

    if not profiles:
        raise ValueError("At least one agent profile is required")
    return profiles
