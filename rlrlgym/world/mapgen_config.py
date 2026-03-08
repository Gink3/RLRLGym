"""Map generation config loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


REQUIRED_MAPGEN_FIELDS = {
    "wall_tile_id",
    "floor_fallback_id",
    "chest_density",
    "min_width",
    "min_height",
}

DEFAULT_RESOURCE_RULES: Dict[str, object] = {
    "water_tile_ids": ["water", "shallow_water", "deep_water"],
    "minable_wall_tile_ids": ["rock_wall", "stone_wall"],
    "minable_ground_tile_ids": ["stone_floor"],
    "tree_tile_ids": ["tree"],
    "forage_tile_ids": ["grass", "bush"],
    "animal_forage_tile_ids": ["grass", "bush", "tree"],
    "harvests_per_tile_limit": 12,
    "tree_chop_progress_required": 10,
    "stone_floor_flint_chance": 0.18,
    "tree_sapling_drop_chance": 0.2,
    "prey_score_hunt_margin": 2,
}

DEFAULT_PLANT_TYPES: Dict[str, Dict[str, object]] = {
    "berry": {
        "seed_item": "berry_seed",
        "crop_tile": "berry_plant",
        "food_item": "berries",
        "food_qty": [1, 3],
        "seed_qty": [1, 2],
    },
    "grain": {
        "seed_item": "grain_seed",
        "crop_tile": "grain_plant",
        "food_item": "grain_bundle",
        "food_qty": [1, 2],
        "seed_qty": [1, 3],
    },
    "herb": {
        "seed_item": "herb_seed",
        "crop_tile": "herb_plant",
        "food_item": "herb_leaf",
        "food_qty": [1, 2],
        "seed_qty": [1, 2],
    },
}


@dataclass
class MapGenConfig:
    wall_tile_id: str = "indestructible_wall"
    floor_fallback_id: str = "floor"
    chest_density: float = 0.02
    monster_density: float = 0.0
    animal_density: float = 0.0
    min_width: int = 4
    min_height: int = 4
    biomes: List[Dict[str, object]] = field(default_factory=list)
    resource_nodes: List[Dict[str, object]] = field(default_factory=list)
    station_spawns: List[Dict[str, object]] = field(default_factory=list)
    structures: List[Dict[str, object]] = field(default_factory=list)
    worldgen: Dict[str, object] = field(default_factory=dict)
    resource_rules: Dict[str, object] = field(default_factory=dict)
    plant_types: Dict[str, Dict[str, object]] = field(default_factory=dict)


def load_mapgen_config(path: str | Path) -> MapGenConfig:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_mapgen_config(raw)


def parse_mapgen_config(raw: object) -> MapGenConfig:
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

    worldgen_raw = row.get("worldgen", {})
    worldgen = dict(worldgen_raw) if isinstance(worldgen_raw, dict) else {}
    resource_rules_raw = row.get("resource_rules", {})
    resource_rules = dict(DEFAULT_RESOURCE_RULES)
    if isinstance(resource_rules_raw, dict):
        resource_rules.update(dict(resource_rules_raw))
    plant_types = {str(k): dict(v) for k, v in DEFAULT_PLANT_TYPES.items()}
    plant_types_raw = row.get("plant_types", {})
    if isinstance(plant_types_raw, dict):
        for key, value in plant_types_raw.items():
            if isinstance(value, dict):
                plant_types[str(key)] = dict(value)

    return MapGenConfig(
        wall_tile_id=str(row["wall_tile_id"]),
        floor_fallback_id=str(row["floor_fallback_id"]),
        chest_density=float(row["chest_density"]),
        monster_density=float(row.get("monster_density", 0.0)),
        animal_density=float(row.get("animal_density", 0.0)),
        min_width=int(row["min_width"]),
        min_height=int(row["min_height"]),
        biomes=[
            dict(x)
            for x in row.get("biomes", [])
            if isinstance(x, dict)
        ],
        resource_nodes=[
            dict(x)
            for x in row.get("resource_nodes", [])
            if isinstance(x, dict)
        ],
        station_spawns=[
            dict(x)
            for x in row.get("station_spawns", [])
            if isinstance(x, dict)
        ],
        structures=[
            dict(x)
            for x in row.get("structures", [])
            if isinstance(x, dict)
        ],
        worldgen=worldgen,
        resource_rules=resource_rules,
        plant_types=plant_types,
    )
