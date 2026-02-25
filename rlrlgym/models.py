"""Data models for environment state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

DEFAULT_SKILL_LEVELS = {
    "melee": 0,
    "archery": 0,
    "thrown_weapons": 0,
    "medic": 0,
    "athletics": 0,
    "exploration": 0,
}


@dataclass
class TileDef:
    tile_id: str
    glyph: str
    color: str
    walkable: bool
    spawn_weight: float
    max_interactions: int
    loot_table: List[str]


@dataclass
class AgentState:
    agent_id: str
    position: Tuple[int, int]
    profile_name: str = "human"
    race_name: str = "human"
    class_name: str = "wanderer"
    hp: int = 10
    max_hp: int = 10
    hunger: int = 20
    max_hunger: int = 20
    inventory: List[str] = field(default_factory=list)
    equipped: List[str] = field(default_factory=list)
    alive: bool = True
    visited: Set[Tuple[int, int]] = field(default_factory=set)
    wait_streak: int = 0
    recent_positions: List[Tuple[int, int]] = field(default_factory=list)
    strength: int = 5
    dexterity: int = 5
    intellect: int = 5
    skills: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SKILL_LEVELS))
    skill_xp: Dict[str, int] = field(default_factory=dict)


@dataclass
class ChestState:
    position: Tuple[int, int]
    opened: bool = False
    locked: bool = False
    loot: List[str] = field(default_factory=list)


@dataclass
class MonsterState:
    entity_id: str
    monster_id: str
    name: str
    symbol: str
    color: str
    position: Tuple[int, int]
    hp: int
    max_hp: int
    acc: int
    eva: int
    dmg_min: int
    dmg_max: int
    dr_min: int
    dr_max: int
    alive: bool = True


@dataclass
class EnvState:
    grid: List[List[str]]
    tile_interactions: Dict[Tuple[int, int], int]
    ground_items: Dict[Tuple[int, int], List[str]]
    agents: Dict[str, AgentState]
    chests: Dict[Tuple[int, int], ChestState] = field(default_factory=dict)
    monsters: Dict[str, MonsterState] = field(default_factory=dict)
    step_count: int = 0
