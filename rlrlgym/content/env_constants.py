"""Load JSON-backed environment constants used by env.py."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class RewardConstants:
    move_valid_reward: float
    move_step_cost: float
    move_food_progress_reward: float
    move_food_regress_penalty: float
    eat_per_hunger_gain_reward: float
    eat_waste_threshold: float
    eat_waste_penalty: float
    low_hunger_threshold: float
    low_hunger_penalty_scale: float


@dataclass(frozen=True)
class CombatConstants:
    damage_type_slash: str
    damage_type_pierce: str
    damage_type_blunt: str
    unarmed_damage_range: Tuple[int, int]
    ring_armor_slots: Tuple[str, ...]
    ring_item_slot: str
    hit_slot_weights: Tuple[Tuple[str, int], ...]
    hit_slot_to_armor_slots: Dict[str, Tuple[str, ...]]
    armor_class_to_skill: Dict[str, str]
    tool_durability_use_by_category: Dict[str, int]


@dataclass(frozen=True)
class WorldConstants:
    profile_aliases: Dict[str, str]
    default_vision_range: int
    fire_fuel_max: int
    fire_fuel_per_stick: int
    fire_fuel_per_wood: int
    fire_fuel_per_log: int
    fire_fuel_decay_per_step: int
    fire_container_tile_ids: frozenset[str]
    default_construct_tile_ids: frozenset[str]
    opaque_tile_ids: frozenset[str]


def _load_json(path: str | Path) -> dict:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object")
    if "schema_version" not in raw or not isinstance(raw["schema_version"], int):
        raise ValueError(f"{path} requires integer schema_version")
    return raw


def load_reward_constants(
    path: str | Path = "data/base/reward_constants.json",
) -> RewardConstants:
    raw = _load_json(path)
    payload = raw.get("reward_constants")
    if not isinstance(payload, dict):
        raise ValueError("reward_constants payload must be an object")
    return RewardConstants(
        move_valid_reward=float(payload["move_valid_reward"]),
        move_step_cost=float(payload["move_step_cost"]),
        move_food_progress_reward=float(payload["move_food_progress_reward"]),
        move_food_regress_penalty=float(payload["move_food_regress_penalty"]),
        eat_per_hunger_gain_reward=float(payload["eat_per_hunger_gain_reward"]),
        eat_waste_threshold=float(payload["eat_waste_threshold"]),
        eat_waste_penalty=float(payload["eat_waste_penalty"]),
        low_hunger_threshold=float(payload["low_hunger_threshold"]),
        low_hunger_penalty_scale=float(payload["low_hunger_penalty_scale"]),
    )


def load_combat_constants(
    path: str | Path = "data/base/combat_constants.json",
) -> CombatConstants:
    raw = _load_json(path)
    payload = raw.get("combat_constants")
    if not isinstance(payload, dict):
        raise ValueError("combat_constants payload must be an object")
    return CombatConstants(
        damage_type_slash=str(payload["damage_type_slash"]),
        damage_type_pierce=str(payload["damage_type_pierce"]),
        damage_type_blunt=str(payload["damage_type_blunt"]),
        unarmed_damage_range=tuple(int(v) for v in payload["unarmed_damage_range"][:2]),
        ring_armor_slots=tuple(str(v) for v in payload["ring_armor_slots"]),
        ring_item_slot=str(payload["ring_item_slot"]),
        hit_slot_weights=tuple(
            (str(row["slot"]), int(row["weight"]))
            for row in payload["hit_slot_weights"]
        ),
        hit_slot_to_armor_slots={
            str(key): tuple(str(v) for v in values)
            for key, values in payload["hit_slot_to_armor_slots"].items()
        },
        armor_class_to_skill={
            str(key): str(value)
            for key, value in payload["armor_class_to_skill"].items()
        },
        tool_durability_use_by_category={
            str(key): int(value)
            for key, value in payload["tool_durability_use_by_category"].items()
        },
    )


def load_world_constants(
    path: str | Path = "data/base/world_constants.json",
) -> WorldConstants:
    raw = _load_json(path)
    payload = raw.get("world_constants")
    if not isinstance(payload, dict):
        raise ValueError("world_constants payload must be an object")
    return WorldConstants(
        profile_aliases={
            str(key): str(value)
            for key, value in payload["profile_aliases"].items()
        },
        default_vision_range=int(payload["default_vision_range"]),
        fire_fuel_max=int(payload["fire_fuel_max"]),
        fire_fuel_per_stick=int(payload["fire_fuel_per_stick"]),
        fire_fuel_per_wood=int(payload["fire_fuel_per_wood"]),
        fire_fuel_per_log=int(payload["fire_fuel_per_log"]),
        fire_fuel_decay_per_step=int(payload["fire_fuel_decay_per_step"]),
        fire_container_tile_ids=frozenset(str(v) for v in payload["fire_container_tile_ids"]),
        default_construct_tile_ids=frozenset(str(v) for v in payload["default_construct_tile_ids"]),
        opaque_tile_ids=frozenset(str(v) for v in payload["opaque_tile_ids"]),
    )
