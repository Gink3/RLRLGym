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
    "mining": 0,
    "woodcutting": 0,
    "crafting": 0,
    "smithing": 0,
    "alchemy": 0,
    "farming": 0,
    "foraging": 0,
    "armor_light": 0,
    "armor_medium": 0,
    "armor_heavy": 0,
}

DEFAULT_ARMOR_SLOTS = {
    "head": None,
    "chest": None,
    "back": None,
    "arms": None,
    "legs": None,
    "neck": None,
    "ring_1": None,
    "ring_2": None,
    "ring_3": None,
    "ring_4": None,
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
    profile_name: str = "reward_explorer_policy_v1"
    race_name: str = "human"
    class_name: str = "fighter"
    hp: int = 10
    max_hp: int = 10
    mana: int = 0
    max_mana: int = 0
    hunger: int = 20
    max_hunger: int = 20
    inventory: List[str] = field(default_factory=list)
    equipped: List[str] = field(default_factory=list)
    armor_slots: Dict[str, str | None] = field(
        default_factory=lambda: dict(DEFAULT_ARMOR_SLOTS)
    )
    faction_id: int = -1
    alive: bool = True
    visited: Set[Tuple[int, int]] = field(default_factory=set)
    wait_streak: int = 0
    recent_positions: List[Tuple[int, int]] = field(default_factory=list)
    strength: int = 5
    dexterity: int = 5
    intellect: int = 5
    skills: Dict[str, int] = field(default_factory=lambda: dict(DEFAULT_SKILL_LEVELS))
    skill_xp: Dict[str, int] = field(default_factory=dict)
    spell_cooldowns: Dict[str, int] = field(default_factory=dict)
    known_spells: List[str] = field(default_factory=list)


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
class ResourceNodeState:
    node_id: str
    position: Tuple[int, int]
    skill: str
    drop_item: str
    remaining: int
    max_yield: int
    biome: str = ""


@dataclass
class StationState:
    station_id: str
    position: Tuple[int, int]
    speed_multiplier: float = 1.0
    quality_tier: int = 0
    unlock_recipes: List[str] = field(default_factory=list)


@dataclass
class ActiveStatus:
    status_id: str
    remaining: int
    tick_interval: int = 1
    tick_counter: int = 0
    source_id: str = ""


@dataclass
class AnimalState:
    entity_id: str
    animal_id: str
    name: str
    symbol: str
    color: str
    position: Tuple[int, int]
    hp: int
    max_hp: int
    hunger: int
    max_hunger: int
    thirst: int
    max_thirst: int
    age: int
    mature_age: int
    reproduction_cooldown: int
    reproduction_cooldown_max: int
    can_shear: bool = False
    sheared: bool = False
    shear_item: str = ""
    wool_regrow: int = 0
    shear_regrow_max: int = 0
    alive: bool = True


@dataclass
class PlantPlotState:
    crop_id: str
    planter_id: str = ""
    planter_faction_id: int = -1


@dataclass
class EnvState:
    grid: List[List[str]]
    tile_interactions: Dict[Tuple[int, int], int]
    tile_harvest_counts: Dict[Tuple[int, int], int]
    ground_items: Dict[Tuple[int, int], List[str]]
    agents: Dict[str, AgentState]
    chests: Dict[Tuple[int, int], ChestState] = field(default_factory=dict)
    monsters: Dict[str, MonsterState] = field(default_factory=dict)
    biomes: Dict[Tuple[int, int], str] = field(default_factory=dict)
    resource_nodes: Dict[Tuple[int, int], ResourceNodeState] = field(default_factory=dict)
    stations: Dict[Tuple[int, int], StationState] = field(default_factory=dict)
    animals: Dict[str, AnimalState] = field(default_factory=dict)
    plant_plots: Dict[Tuple[int, int], PlantPlotState] = field(default_factory=dict)
    agent_statuses: Dict[str, List[ActiveStatus]] = field(default_factory=dict)
    item_metadata: Dict[str, Dict[str, object]] = field(default_factory=dict)
    faction_leaders: Dict[int, str] = field(default_factory=dict)
    pending_faction_invites: Dict[str, Dict[str, int | str]] = field(default_factory=dict)
    step_count: int = 0
