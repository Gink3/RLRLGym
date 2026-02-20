"""Procedural map generation."""

from __future__ import annotations

import random
from typing import Dict, List, Tuple

from .models import TileDef
from .tiles import weighted_tile_ids, weighted_tile_weights


def generate_map(
    width: int,
    height: int,
    tiles: Dict[str, TileDef],
    rng: random.Random,
    wall_tile_id: str = "wall",
    floor_fallback_id: str = "floor",
) -> List[List[str]]:
    if width < 4 or height < 4:
        raise ValueError("Map must be at least 4x4")

    interior_ids = weighted_tile_ids(tiles)
    interior_weights = weighted_tile_weights(tiles)

    if not interior_ids:
        raise ValueError("No tiles available for map generation")

    grid: List[List[str]] = []
    for r in range(height):
        row: List[str] = []
        for c in range(width):
            if r in (0, height - 1) or c in (0, width - 1):
                row.append(wall_tile_id if wall_tile_id in tiles else floor_fallback_id)
            else:
                row.append(rng.choices(interior_ids, weights=interior_weights, k=1)[0])
        grid.append(row)

    # Ensure enough walkable floor to place agents.
    walkable = [(r, c) for r in range(height) for c in range(width) if tiles[grid[r][c]].walkable]
    if len(walkable) < 2:
        for r in range(1, height - 1):
            for c in range(1, width - 1):
                grid[r][c] = floor_fallback_id

    return grid


def sample_walkable_positions(
    grid: List[List[str]], tiles: Dict[str, TileDef], count: int, rng: random.Random
) -> List[Tuple[int, int]]:
    walkable = [(r, c) for r, row in enumerate(grid) for c, tile_id in enumerate(row) if tiles[tile_id].walkable]
    if len(walkable) < count:
        raise ValueError("Not enough walkable cells for agent placement")
    rng.shuffle(walkable)
    return walkable[:count]
