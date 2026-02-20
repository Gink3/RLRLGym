"""Tile loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .models import TileDef


def load_tileset(path: str | Path) -> Dict[str, TileDef]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    tiles: Dict[str, TileDef] = {}
    for row in data:
        tile = TileDef(
            tile_id=row["id"],
            glyph=row["glyph"],
            color=row.get("color", "white"),
            walkable=bool(row.get("walkable", True)),
            spawn_weight=float(row.get("spawn_weight", 1.0)),
            max_interactions=int(row.get("max_interactions", 0)),
            loot_table=list(row.get("loot_table", [])),
        )
        tiles[tile.tile_id] = tile
    if not tiles:
        raise ValueError("Tileset is empty")
    return tiles


def weighted_tile_ids(tiles: Dict[str, TileDef]) -> List[str]:
    return [tile.tile_id for tile in tiles.values() if tile.spawn_weight > 0]


def weighted_tile_weights(tiles: Dict[str, TileDef]) -> List[float]:
    return [tile.spawn_weight for tile in tiles.values() if tile.spawn_weight > 0]
