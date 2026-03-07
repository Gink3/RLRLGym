"""Agent race definitions and JSON loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict


REQUIRED_RACE_FIELDS = {
    "name",
    "strength",
    "dexterity",
    "intellect",
    "base_dr_min",
    "base_dr_max",
    "dr_bonus_vs",
}


@dataclass
class AgentRace:
    name: str
    strength: int
    dexterity: int
    intellect: int
    base_dr_min: int = 0
    base_dr_max: int = 1
    dr_bonus_vs: Dict[str, int] = field(default_factory=dict)


def load_races(path: str | Path) -> Dict[str, AgentRace]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Agent races JSON must be an object")
    if "schema_version" not in raw or not isinstance(raw["schema_version"], int):
        raise ValueError("Agent races JSON requires integer schema_version")
    if "races" not in raw or not isinstance(raw["races"], list):
        raise ValueError("Agent races JSON requires array 'races'")

    races: Dict[str, AgentRace] = {}
    for idx, row in enumerate(raw["races"]):
        if not isinstance(row, dict):
            raise ValueError(f"race[{idx}] must be an object")
        missing = REQUIRED_RACE_FIELDS - set(row.keys())
        if missing:
            miss = ", ".join(sorted(missing))
            raise ValueError(f"race[{idx}] missing required field(s): {miss}")
        if not isinstance(row["dr_bonus_vs"], dict):
            raise ValueError(f"race[{idx}].dr_bonus_vs must be an object")

        race = AgentRace(
            name=str(row["name"]),
            strength=int(row["strength"]),
            dexterity=int(row["dexterity"]),
            intellect=int(row["intellect"]),
            base_dr_min=int(row["base_dr_min"]),
            base_dr_max=int(row["base_dr_max"]),
            dr_bonus_vs={str(k): int(v) for k, v in row["dr_bonus_vs"].items()},
        )
        races[race.name] = race

    if not races:
        raise ValueError("At least one agent race is required")
    return races
