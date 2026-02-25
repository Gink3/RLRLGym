"""Monster definition and spawn table loading helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


REQUIRED_MONSTER_FIELDS = {
    "id",
    "name",
    "symbol",
    "color",
    "threat",
    "hp",
    "acc",
    "eva",
    "dmg_min",
    "dmg_max",
    "dr_min",
    "dr_max",
    "loot",
}
REQUIRED_LOOT_FIELDS = {"item", "weight", "min_qty", "max_qty"}
REQUIRED_SPAWN_FIELDS = {"monster_id", "weight"}


@dataclass
class MonsterLootEntry:
    item: str
    weight: float
    min_qty: int = 1
    max_qty: int = 1


@dataclass
class MonsterDef:
    monster_id: str
    name: str
    symbol: str
    color: str
    threat: int
    hp: int
    acc: int
    eva: int
    dmg_min: int
    dmg_max: int
    dr_min: int = 0
    dr_max: int = 0
    loot: List[MonsterLootEntry] = field(default_factory=list)


@dataclass
class MonsterSpawnEntry:
    monster_id: str
    weight: float


def load_monsters(path: str | Path) -> Dict[str, MonsterDef]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Monsters JSON must be an object")
    if "schema_version" not in raw or not isinstance(raw["schema_version"], int):
        raise ValueError("Monsters JSON requires integer schema_version")
    if "monsters" not in raw or not isinstance(raw["monsters"], list):
        raise ValueError("Monsters JSON requires array 'monsters'")

    monsters: Dict[str, MonsterDef] = {}
    symbol_color_pairs: set[tuple[str, str]] = set()
    for idx, row in enumerate(raw["monsters"]):
        if not isinstance(row, dict):
            raise ValueError(f"monster[{idx}] must be an object")
        missing = REQUIRED_MONSTER_FIELDS - set(row.keys())
        if missing:
            miss = ", ".join(sorted(missing))
            raise ValueError(f"monster[{idx}] missing required field(s): {miss}")
        if not isinstance(row["loot"], list):
            raise ValueError(f"monster[{idx}].loot must be an array")
        symbol = str(row["symbol"])
        color = str(row["color"])
        if len(symbol) != 1:
            raise ValueError(f"monster[{idx}].symbol must be a single character")

        pair = (symbol, color)
        if pair in symbol_color_pairs:
            raise ValueError(
                f"monster[{idx}] duplicates symbol/color combination {pair}"
            )
        symbol_color_pairs.add(pair)

        loot: List[MonsterLootEntry] = []
        for lidx, entry in enumerate(row["loot"]):
            if not isinstance(entry, dict):
                raise ValueError(f"monster[{idx}].loot[{lidx}] must be an object")
            lmissing = REQUIRED_LOOT_FIELDS - set(entry.keys())
            if lmissing:
                miss = ", ".join(sorted(lmissing))
                raise ValueError(
                    f"monster[{idx}].loot[{lidx}] missing required field(s): {miss}"
                )
            loot.append(
                MonsterLootEntry(
                    item=str(entry["item"]),
                    weight=float(entry["weight"]),
                    min_qty=int(entry["min_qty"]),
                    max_qty=int(entry["max_qty"]),
                )
            )

        monster = MonsterDef(
            monster_id=str(row["id"]),
            name=str(row["name"]),
            symbol=symbol,
            color=color,
            threat=int(row["threat"]),
            hp=int(row["hp"]),
            acc=int(row["acc"]),
            eva=int(row["eva"]),
            dmg_min=int(row["dmg_min"]),
            dmg_max=int(row["dmg_max"]),
            dr_min=int(row["dr_min"]),
            dr_max=int(row["dr_max"]),
            loot=loot,
        )
        monsters[monster.monster_id] = monster

    if not monsters:
        raise ValueError("At least one monster is required")
    return monsters


def load_monster_spawns(
    path: str | Path, monsters: Dict[str, MonsterDef]
) -> List[MonsterSpawnEntry]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Monster spawn JSON must be an object")
    if "schema_version" not in raw or not isinstance(raw["schema_version"], int):
        raise ValueError("Monster spawn JSON requires integer schema_version")
    if "spawns" not in raw or not isinstance(raw["spawns"], list):
        raise ValueError("Monster spawn JSON requires array 'spawns'")

    out: List[MonsterSpawnEntry] = []
    seen: set[str] = set()
    for idx, row in enumerate(raw["spawns"]):
        if not isinstance(row, dict):
            raise ValueError(f"spawns[{idx}] must be an object")
        missing = REQUIRED_SPAWN_FIELDS - set(row.keys())
        if missing:
            miss = ", ".join(sorted(missing))
            raise ValueError(f"spawns[{idx}] missing required field(s): {miss}")
        monster_id = str(row["monster_id"])
        if monster_id not in monsters:
            raise ValueError(f"spawns[{idx}] references unknown monster_id '{monster_id}'")
        if monster_id in seen:
            raise ValueError(f"spawns[{idx}] duplicates monster_id '{monster_id}'")
        seen.add(monster_id)
        out.append(
            MonsterSpawnEntry(
                monster_id=monster_id,
                weight=float(row["weight"]),
            )
        )

    if not out:
        raise ValueError("At least one spawn entry is required")
    return out
