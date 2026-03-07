"""Static map layout loading and validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class StaticMapLayout:
    grid: List[List[str]]
    biomes: Dict[Tuple[int, int], str]
    name: str = ""


def load_map_layout(path: str | Path) -> StaticMapLayout:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_map_layout(raw)


def parse_map_layout(raw: object) -> StaticMapLayout:
    if not isinstance(raw, dict):
        raise ValueError("Map layout JSON must be an object")
    if "schema_version" not in raw or not isinstance(raw["schema_version"], int):
        raise ValueError("Map layout JSON requires integer schema_version")
    body = raw.get("map", raw)
    if not isinstance(body, dict):
        raise ValueError("Map layout payload must be an object")
    grid = body.get("grid")
    if not isinstance(grid, list) or not grid:
        raise ValueError("map.grid must be a non-empty 2D array")
    width = -1
    norm_grid: List[List[str]] = []
    for ridx, row in enumerate(grid):
        if not isinstance(row, list) or not row:
            raise ValueError(f"map.grid[{ridx}] must be a non-empty array")
        norm_row = [str(x) for x in row]
        if width < 0:
            width = len(norm_row)
        elif len(norm_row) != width:
            raise ValueError("map.grid rows must all have equal length")
        norm_grid.append(norm_row)
    biomes: Dict[Tuple[int, int], str] = {}
    for idx, row in enumerate(body.get("biomes", [])):
        if not isinstance(row, dict):
            raise ValueError(f"map.biomes[{idx}] must be an object")
        pos = row.get("position", [])
        if not isinstance(pos, list) or len(pos) != 2:
            raise ValueError(f"map.biomes[{idx}].position must be [r, c]")
        r = int(pos[0])
        c = int(pos[1])
        if r < 0 or c < 0 or r >= len(norm_grid) or c >= width:
            raise ValueError(f"map.biomes[{idx}] position out of bounds: {(r, c)}")
        biomes[(r, c)] = str(row.get("biome", ""))
    return StaticMapLayout(
        grid=norm_grid,
        biomes=biomes,
        name=str(body.get("name", "")),
    )
