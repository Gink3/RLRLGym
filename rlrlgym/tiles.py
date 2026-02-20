"""Tile loading helpers with schema validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .models import TileDef

REQUIRED_TILE_FIELDS = {
    "id",
    "glyph",
    "color",
    "walkable",
    "spawn_weight",
    "max_interactions",
    "loot_table",
}


def _validate_tile_schema(data: object) -> dict:
    if not isinstance(data, dict):
        raise ValueError(
            "Tile JSON must be an object with 'schema_version' and 'tiles'"
        )

    if "schema_version" not in data:
        raise ValueError("Tile JSON missing required key: schema_version")
    if not isinstance(data["schema_version"], int):
        raise ValueError("schema_version must be an integer")

    if "tiles" not in data:
        raise ValueError("Tile JSON missing required key: tiles")
    if not isinstance(data["tiles"], list):
        raise ValueError("tiles must be a list")

    for idx, row in enumerate(data["tiles"]):
        if not isinstance(row, dict):
            raise ValueError(f"tile[{idx}] must be an object")
        missing = REQUIRED_TILE_FIELDS - set(row.keys())
        if missing:
            missing_sorted = ", ".join(sorted(missing))
            raise ValueError(f"tile[{idx}] missing required field(s): {missing_sorted}")
        if not isinstance(row["loot_table"], list):
            raise ValueError(f"tile[{idx}].loot_table must be an array")
        if not isinstance(row["glyph"], str) or len(row["glyph"]) != 1:
            raise ValueError(f"tile[{idx}].glyph must be a single character")

    return data


def load_tileset(path: str | Path) -> Dict[str, TileDef]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    data = _validate_tile_schema(raw)

    tiles: Dict[str, TileDef] = {}
    for row in data["tiles"]:
        tile = TileDef(
            tile_id=row["id"],
            glyph=row["glyph"],
            color=row["color"],
            walkable=bool(row["walkable"]),
            spawn_weight=float(row["spawn_weight"]),
            max_interactions=int(row["max_interactions"]),
            loot_table=list(row["loot_table"]),
        )
        tiles[tile.tile_id] = tile

    if not tiles:
        raise ValueError("Tileset is empty")
    return tiles


def weighted_tile_ids(tiles: Dict[str, TileDef]) -> List[str]:
    return [tile.tile_id for tile in tiles.values() if tile.spawn_weight > 0]


def weighted_tile_weights(tiles: Dict[str, TileDef]) -> List[float]:
    return [tile.spawn_weight for tile in tiles.values() if tile.spawn_weight > 0]
