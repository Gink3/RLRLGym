"""Animal definition loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class AnimalDef:
    animal_id: str
    name: str
    symbol: str
    color: str
    spawn_weight: float
    hp: int
    drop_item: str
    max_hunger: int
    max_thirst: int
    movement_speed: int
    prey_score: int
    carnivore: bool
    litter_size_min: int
    litter_size_max: int
    mature_age: int
    reproduction_cooldown: int
    can_shear: bool = False
    shear_item: str = ""
    shear_regrow_steps: int = 0


def load_animals(path: str | Path) -> Dict[str, AnimalDef]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_animals(raw)


def parse_animals(raw: object) -> Dict[str, AnimalDef]:
    if not isinstance(raw, dict):
        raise ValueError("Animals JSON must be an object")
    if "schema_version" not in raw or not isinstance(raw["schema_version"], int):
        raise ValueError("Animals JSON requires integer schema_version")
    if "animals" not in raw or not isinstance(raw["animals"], list):
        raise ValueError("Animals JSON requires array 'animals'")
    out: Dict[str, AnimalDef] = {}
    for idx, row in enumerate(raw["animals"]):
        if not isinstance(row, dict):
            raise ValueError(f"animals[{idx}] must be an object")
        animal_id = str(row.get("id", "")).strip()
        if not animal_id:
            raise ValueError(f"animals[{idx}] missing required field 'id'")
        out[animal_id] = AnimalDef(
            animal_id=animal_id,
            name=str(row.get("name", animal_id)),
            symbol=(str(row.get("symbol", "a"))[:1] or "a"),
            color=str(row.get("color", "green")),
            spawn_weight=max(0.0, float(row.get("spawn_weight", 1.0))),
            hp=max(1, int(row.get("hp", 4))),
            drop_item=str(row.get("drop_item", "")).strip(),
            max_hunger=max(2, int(row.get("max_hunger", max(3, int(row.get("hp", 4)) + 4)))),
            max_thirst=max(2, int(row.get("max_thirst", max(3, int(row.get("hp", 4)) + 4)))),
            movement_speed=max(1, int(row.get("movement_speed", 1))),
            prey_score=max(0, int(row.get("prey_score", 1))),
            carnivore=bool(row.get("carnivore", False)),
            litter_size_min=max(1, int(row.get("litter_size_min", 1))),
            litter_size_max=max(1, int(row.get("litter_size_max", row.get("litter_size_min", 1)))),
            mature_age=max(1, int(row.get("mature_age", 8))),
            reproduction_cooldown=max(1, int(row.get("reproduction_cooldown", 8))),
            can_shear=bool(row.get("can_shear", False)),
            shear_item=str(row.get("shear_item", "")).strip(),
            shear_regrow_steps=max(1, int(row.get("shear_regrow_steps", 6))),
        )
        if out[animal_id].litter_size_max < out[animal_id].litter_size_min:
            out[animal_id].litter_size_max = out[animal_id].litter_size_min
    return out
