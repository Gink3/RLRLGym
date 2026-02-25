"""Agent class definitions and JSON loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


REQUIRED_CLASS_FIELDS = {
    "name",
    "starting_items",
    "skill_modifiers",
}


@dataclass
class AgentClass:
    name: str
    starting_items: List[str] = field(default_factory=list)
    skill_modifiers: Dict[str, int] = field(default_factory=dict)


def load_classes(path: str | Path) -> Dict[str, AgentClass]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Agent classes JSON must be an object")
    if "schema_version" not in raw or not isinstance(raw["schema_version"], int):
        raise ValueError("Agent classes JSON requires integer schema_version")
    if "classes" not in raw or not isinstance(raw["classes"], list):
        raise ValueError("Agent classes JSON requires array 'classes'")

    classes: Dict[str, AgentClass] = {}
    for idx, row in enumerate(raw["classes"]):
        if not isinstance(row, dict):
            raise ValueError(f"class[{idx}] must be an object")
        missing = REQUIRED_CLASS_FIELDS - set(row.keys())
        if missing:
            miss = ", ".join(sorted(missing))
            raise ValueError(f"class[{idx}] missing required field(s): {miss}")
        if not isinstance(row["starting_items"], list):
            raise ValueError(f"class[{idx}].starting_items must be an array")
        if not isinstance(row["skill_modifiers"], dict):
            raise ValueError(f"class[{idx}].skill_modifiers must be an object")

        cls = AgentClass(
            name=str(row["name"]),
            starting_items=[str(x) for x in row["starting_items"]],
            skill_modifiers={str(k): int(v) for k, v in row["skill_modifiers"].items()},
        )
        classes[cls.name] = cls

    if not classes:
        raise ValueError("At least one agent class is required")
    return classes
