"""Map generation config loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


REQUIRED_MAPGEN_FIELDS = {
    "wall_tile_id",
    "floor_fallback_id",
    "chest_density",
    "min_width",
    "min_height",
}


@dataclass
class MapGenConfig:
    wall_tile_id: str = "wall"
    floor_fallback_id: str = "floor"
    chest_density: float = 0.02
    monster_density: float = 0.0
    min_width: int = 4
    min_height: int = 4


def load_mapgen_config(path: str | Path) -> MapGenConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Mapgen config JSON must be an object")
    if "schema_version" not in raw or not isinstance(raw["schema_version"], int):
        raise ValueError("Mapgen config JSON requires integer schema_version")
    if "mapgen" not in raw or not isinstance(raw["mapgen"], dict):
        raise ValueError("Mapgen config JSON requires object 'mapgen'")

    row = raw["mapgen"]
    missing = REQUIRED_MAPGEN_FIELDS - set(row.keys())
    if missing:
        miss = ", ".join(sorted(missing))
        raise ValueError(f"mapgen missing required field(s): {miss}")

    return MapGenConfig(
        wall_tile_id=str(row["wall_tile_id"]),
        floor_fallback_id=str(row["floor_fallback_id"]),
        chest_density=float(row["chest_density"]),
        monster_density=float(row.get("monster_density", 0.0)),
        min_width=int(row["min_width"]),
        min_height=int(row["min_height"]),
    )
