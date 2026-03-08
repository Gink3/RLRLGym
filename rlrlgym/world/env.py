"""PettingZoo-style parallel multi-agent roguelike environment."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from ..systems.constants import (
    ACTION_MAX,
    ACTION_ACCEPT_INVITE,
    ACTION_ATTACK,
    ACTION_DEFEND,
    ACTION_EAT,
    ACTION_EQUIP,
    ACTION_GIVE,
    ACTION_GUARD,
    ACTION_INTERACT,
    ACTION_LEAVE_FACTION,
    ACTION_LOOT,
    ACTION_MOVE_EAST,
    ACTION_MOVE_NORTH,
    ACTION_MOVE_SOUTH,
    ACTION_MOVE_WEST,
    ACTION_TRADE,
    ACTION_PICKUP,
    ACTION_REVIVE,
    ACTION_INTERACT_CHEST,
    ACTION_INTERACT_DRINK,
    ACTION_INTERACT_FACTION,
    ACTION_INTERACT_FORAGE,
    ACTION_INTERACT_HARVEST,
    ACTION_INTERACT_MINE,
    ACTION_INTERACT_PLANT,
    ACTION_INTERACT_RESOURCE,
    ACTION_INTERACT_SHEAR,
    ACTION_INTERACT_SHRINE,
    ACTION_INTERACT_STATION,
    ACTION_USE,
    ACTION_WAIT,
    MOVE_DELTAS,
)
from ..content.classes import AgentClass, load_classes
from ..content.animals import AnimalDef, load_animals, parse_animals
from ..content.items import ItemCatalog, load_items, parse_items
from .mapgen import generate_biome_terrain, generate_map, sample_walkable_positions
from .mapgen_config import MapGenConfig, load_mapgen_config, parse_mapgen_config
from .map_layout import StaticMapLayout, load_map_layout, parse_map_layout
from ..systems.models import (
    ActiveStatus,
    AgentState,
    AnimalState,
    ChestState,
    EnvState,
    MonsterState,
    PlantPlotState,
    ResourceNodeState,
    StationState,
)
from ..content.enchantments import EnchantDef, load_enchantments, parse_enchantments
from ..content.monsters import (
    MonsterDef,
    MonsterSpawnEntry,
    load_monster_spawns,
    load_monsters,
    parse_monster_spawns,
    parse_monsters,
)
from ..content.profiles import AgentProfile, load_profiles
from ..content.races import AgentRace, load_races
from ..content.recipes import RecipeDef, load_recipes, parse_recipes
from ..content.spells import SpellDef, load_spells, parse_spells
from ..content.statuses import StatusDef, load_statuses, parse_statuses
from .render import RenderWindow
from ..systems.scenario import apply_scenario_to_env_config, load_scenario
from ..content.structures import load_structures_config, parse_structures_config
from ..content.tiles import load_tileset, parse_tileset

MOVE_VALID_REWARD = 0.005
MOVE_STEP_COST = 0.002
MOVE_FOOD_PROGRESS_REWARD = 0.01
MOVE_FOOD_REGRESS_PENALTY = 0.005
EAT_PER_HUNGER_GAIN_REWARD = 0.06
EAT_WASTE_THRESHOLD = 0.8
EAT_WASTE_PENALTY = 0.05
LOW_HUNGER_THRESHOLD = 0.4
LOW_HUNGER_PENALTY_SCALE = 0.006

DAMAGE_TYPE_SLASH = "slash"
DAMAGE_TYPE_PIERCE = "pierce"
DAMAGE_TYPE_BLUNT = "blunt"

UNARMED_DAMAGE_RANGE = (1, 2)
RING_ARMOR_SLOTS = ("ring_1", "ring_2", "ring_3", "ring_4")
RING_ITEM_SLOT = "ring"
HIT_SLOT_WEIGHTS: Tuple[Tuple[str, int], ...] = (
    ("head", 14),
    ("chest", 30),
    ("back", 14),
    ("arms", 14),
    ("legs", 18),
    ("neck", 6),
    ("rings", 4),
)
HIT_SLOT_TO_ARMOR_SLOTS: Dict[str, Tuple[str, ...]] = {
    "head": ("head",),
    "chest": ("chest",),
    "back": ("back",),
    "arms": ("arms",),
    "legs": ("legs",),
    "neck": ("neck",),
    "rings": RING_ARMOR_SLOTS,
}
ARMOR_CLASS_TO_SKILL: Dict[str, str] = {
    "light": "armor_light",
    "medium": "armor_medium",
    "heavy": "armor_heavy",
}
PROFILE_ALIASES: Dict[str, str] = {
    "human": "reward_explorer_policy_v1",
    "orc": "reward_brawler_policy_v1",
}
WATER_TILE_IDS = {"water", "shallow_water", "deep_water"}
MINABLE_WALL_TILE_IDS = {"rock_wall", "stone_wall"}
MINABLE_GROUND_TILE_IDS = {"stone_floor"}
TREE_TILE_IDS = {"tree"}
HARVESTS_PER_TILE_LIMIT = 12
TREE_CHOP_PROGRESS_REQUIRED = 10
STONE_FLOOR_FLINT_CHANCE = 0.18
TREE_SAPLING_DROP_CHANCE = 0.2
DEFAULT_VISION_RANGE = 20
PREY_SCORE_HUNT_MARGIN = 2
FIRE_FUEL_MAX = 20
FIRE_FUEL_PER_STICK = 2
FIRE_FUEL_PER_WOOD = 5
FIRE_FUEL_PER_LOG = 8
FIRE_FUEL_DECAY_PER_STEP = 1
OPAQUE_TILE_IDS = {
    "wall",
    "indestructible_wall",
    "wood_wall",
    "rock_wall",
    "stone_wall",
    "tree",
}
TOOL_DURABILITY_USE_BY_CATEGORY: Dict[str, int] = {
    "axe": 1,
    "pickaxe": 1,
    "shovel": 1,
    "handaxe": 1,
    "knife": 1,
}
PLANT_TYPES: Dict[str, Dict[str, object]] = {
    "berry": {
        "seed_item": "berry_seed",
        "crop_tile": "berry_plant",
        "food_item": "berries",
        "food_qty": (1, 3),
        "seed_qty": (1, 2),
    },
    "grain": {
        "seed_item": "grain_seed",
        "crop_tile": "grain_plant",
        "food_item": "grain_bundle",
        "food_qty": (1, 2),
        "seed_qty": (1, 3),
    },
    "herb": {
        "seed_item": "herb_seed",
        "crop_tile": "herb_plant",
        "food_item": "herb_leaf",
        "food_qty": (1, 2),
        "seed_qty": (1, 2),
    },
}

@dataclass
class EnvConfig:
    width: int = 50
    height: int = 50
    max_steps: int = 150
    n_agents: int = 2
    tiles_path: str = str(Path("data") / "base" / "tiles.json")
    profiles_path: str = str(Path("data") / "base" / "agent" / "agent_profiles.json")
    races_path: str = str(Path("data") / "base" / "agent" / "agent_races.json")
    classes_path: str = str(Path("data") / "base" / "agent" / "agent_classes.json")
    items_path: str = str(Path("data") / "base" / "items.json")
    monsters_path: str = str(Path("data") / "base" / "monsters.json")
    animals_path: str = str(Path("data") / "base" / "animals.json")
    monster_spawns_path: str = str(Path("data") / "base" / "monster_spawns.json")
    mapgen_config_path: str = str(Path("data") / "base" / "mapgen_config.json")
    map_structures_path: str = str(Path("data") / "base" / "structures.json")
    static_map_path: str = ""
    recipes_path: str = str(Path("data") / "base" / "recipes.json")
    statuses_path: str = str(Path("data") / "base" / "statuses.json")
    spells_path: str = str(Path("data") / "base" / "spells.json")
    enchantments_path: str = str(Path("data") / "base" / "enchantments.json")
    # Optional scenario-bundled payloads. When set, these override *_path files.
    structures_data: Dict[str, object] = field(default_factory=dict)
    tiles_data: Dict[str, object] = field(default_factory=dict)
    items_data: Dict[str, object] = field(default_factory=dict)
    monsters_data: Dict[str, object] = field(default_factory=dict)
    animals_data: Dict[str, object] = field(default_factory=dict)
    monster_spawns_data: Dict[str, object] = field(default_factory=dict)
    mapgen_config_data: Dict[str, object] = field(default_factory=dict)
    map_structures_data: Dict[str, object] = field(default_factory=dict)
    static_map_data: Dict[str, object] = field(default_factory=dict)
    recipes_data: Dict[str, object] = field(default_factory=dict)
    statuses_data: Dict[str, object] = field(default_factory=dict)
    spells_data: Dict[str, object] = field(default_factory=dict)
    enchantments_data: Dict[str, object] = field(default_factory=dict)
    scenario_path: str = ""
    agent_scenario: List[Dict[str, object]] = field(default_factory=list)
    agent_observation_config: Dict[str, Dict[str, object]] = field(default_factory=dict)
    agent_profile_map: Dict[str, str] = field(default_factory=dict)
    agent_race_map: Dict[str, str] = field(default_factory=dict)
    agent_class_map: Dict[str, str] = field(default_factory=dict)
    monster_sight_range: int = 7
    combat_training_mode: bool = False
    hunger_tick_enabled: bool = True
    missed_attack_opportunity_penalty: float = 0.03
    # Exploration/search shaping (JSON-configurable).
    new_tile_seen_reward: float = 0.02
    frontier_step_reward: float = 0.008
    stagnation_penalty: float = 0.01
    stagnation_threshold_steps: int = 10
    repeat_visit_penalty: float = 0.006
    repeat_visit_window: int = 6
    move_bias_reward: float = 0.002
    wait_no_enemy_penalty: float = 0.01
    wait_safe_hunger_ratio: float = 0.5
    first_enemy_seen_bonus: float = 0.7
    enemy_visible_reward: float = 0.01
    enemy_distance_delta_reward_scale: float = 0.01
    enemy_distance_delta_clip: float = 2.0
    lost_enemy_penalty: float = 0.01
    timeout_tie_penalty: float = 0.2
    engagement_bonus: float = 0.15
    team_create_reward: float = 0.06
    team_invite_reward: float = 0.03
    team_join_reward: float = 0.14
    team_join_inviter_reward: float = 0.04
    team_join_invitee_reward: float = 0.04
    team_proximity_reward: float = 0.01
    team_action_bonus: float = 0.015
    team_give_reward: float = 0.04
    team_trade_reward: float = 0.05
    team_revive_reward: float = 0.2
    team_guard_reward: float = 0.03
    team_leave_reward: float = 0.01
    faction_invite_ttl_steps: int = 5
    faction_action_cooldown_steps: int = 1
    team_pair_reward_cap_per_episode: int = 8
    team_pair_reward_repeat_guard_steps: int = 2
    team_proximity_pair_cap_per_episode: int = 25
    team_proximity_repeat_guard_steps: int = 1
    guard_damage_reduction_ratio: float = 0.5
    formation_dr_bonus: int = 1
    formation_hit_bonus: float = 0.04
    leader_team_share: float = 0.15
    ally_damage_penalty_per_hp: float = 0.08
    ally_kill_penalty: float = 1.2
    treasure_hold_reward_per_turn: float = 0.01
    treasure_hold_reward_cap_items: int = 6
    treasure_end_bonus_per_item: float = 0.15
    invalid_faction_action_penalty: float = 0.03
    defend_unarmed_dr_bonus: int = 2
    render_enabled: bool = True
    vision_range_default: int = DEFAULT_VISION_RANGE
    enable_los: bool = True

    @classmethod
    def from_json(cls, path: str | Path) -> "EnvConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Env config JSON must be an object")
        if "schema_version" not in raw or not isinstance(raw["schema_version"], int):
            raise ValueError("Env config JSON requires integer schema_version")
        data = raw.get("env_config", raw)
        if not isinstance(data, dict):
            raise ValueError("Env config payload must be an object")

        base = cls()
        merged = dict(base.__dict__)
        for key, value in data.items():
            if key in merged:
                merged[key] = value
        merged["agent_observation_config"] = dict(
            merged.get("agent_observation_config", {})
        )
        merged["agent_scenario"] = list(merged.get("agent_scenario", []))
        merged["agent_profile_map"] = dict(merged.get("agent_profile_map", {}))
        merged["agent_race_map"] = dict(merged.get("agent_race_map", {}))
        merged["agent_class_map"] = dict(merged.get("agent_class_map", {}))
        merged["structures_data"] = dict(merged.get("structures_data", {}) or {})
        merged["tiles_data"] = dict(merged.get("tiles_data", {}) or {})
        merged["items_data"] = dict(merged.get("items_data", {}) or {})
        merged["monsters_data"] = dict(merged.get("monsters_data", {}) or {})
        merged["animals_data"] = dict(merged.get("animals_data", {}) or {})
        merged["monster_spawns_data"] = dict(
            merged.get("monster_spawns_data", {}) or {}
        )
        merged["mapgen_config_data"] = dict(
            merged.get("mapgen_config_data", {}) or {}
        )
        merged["map_structures_data"] = dict(
            merged.get("map_structures_data", {}) or {}
        )
        merged["static_map_data"] = dict(merged.get("static_map_data", {}) or {})
        merged["recipes_data"] = dict(merged.get("recipes_data", {}) or {})
        merged["statuses_data"] = dict(merged.get("statuses_data", {}) or {})
        merged["spells_data"] = dict(merged.get("spells_data", {}) or {})
        merged["enchantments_data"] = dict(merged.get("enchantments_data", {}) or {})
        return cls(**merged)


class MultiAgentRLRLGym:
    """PettingZoo Parallel-like API for multi-agent training.

    `reset(seed, options)` -> observations, info
    `step(actions)` -> observations, rewards, terminations, truncations, info
    """

    metadata = {"name": "RLRLGym-v0", "render_modes": ["window"]}

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        if config is not None:
            self.config = config
        else:
            default_cfg_path = Path("data") / "env_config.json"
            if default_cfg_path.exists():
                self.config = EnvConfig.from_json(default_cfg_path)
            else:
                self.config = EnvConfig()
        self.scenario = None
        if str(self.config.scenario_path).strip():
            self.scenario = load_scenario(self.config.scenario_path)
            self.config = apply_scenario_to_env_config(self.config, self.scenario)
        tiles_payload = (
            self.config.structures_data
            if self.config.structures_data
            else self.config.tiles_data
        )
        if tiles_payload:
            self.tiles = parse_tileset(tiles_payload)
        else:
            self.tiles = load_tileset(self.config.tiles_path)
        self.profiles: Dict[str, AgentProfile] = load_profiles(
            self.config.profiles_path
        )
        self.races: Dict[str, AgentRace] = load_races(self.config.races_path)
        self.classes: Dict[str, AgentClass] = load_classes(self.config.classes_path)
        if self.config.items_data:
            self.items = parse_items(self.config.items_data)
        else:
            self.items = load_items(self.config.items_path)
        if self.config.monsters_data:
            self.monsters = parse_monsters(self.config.monsters_data)
        else:
            self.monsters = load_monsters(self.config.monsters_path)
        if self.config.animals_data:
            self.animals = parse_animals(self.config.animals_data)
        else:
            self.animals = load_animals(self.config.animals_path)
        if self.config.monster_spawns_data:
            self.monster_spawns = parse_monster_spawns(
                self.config.monster_spawns_data, self.monsters
            )
        else:
            self.monster_spawns = load_monster_spawns(
                self.config.monster_spawns_path, self.monsters
            )
        self.weapon_damage_type = dict(self.items.weapon_damage_type)
        self.weapon_damage_range = dict(self.items.weapon_damage_range)
        self.weapon_skill_by_item = dict(self.items.weapon_skill)
        self.weapon_defense_dr_bonus = dict(self.items.weapon_defense_dr_bonus)
        self.weapon_two_handed = dict(self.items.weapon_two_handed)
        self.item_dr_bonus_vs = dict(self.items.item_dr_bonus_vs)
        self.item_defense_dr_bonus = dict(self.items.item_defense_dr_bonus)
        self.tool_category_by_item = dict(self.items.tool_category_by_item)
        self.tool_skill_by_item = dict(self.items.tool_skill_by_item)
        self.tool_skill_bonus_by_item = dict(self.items.tool_skill_bonus_by_item)
        self.base_durability_by_item = dict(self.items.base_durability_by_item)
        self.armor_slot_by_item = dict(self.items.armor_slot_by_item)
        self.armor_class_by_item = dict(self.items.armor_class_by_item)
        self.item_weight = dict(self.items.item_weight)
        self.edible_items = set(self.items.edible_items)
        self.treasure_items = set(self.items.treasure_items)
        self.chest_loot_table = list(self.items.chest_loot_table)
        self._validate_item_references()
        if self.config.mapgen_config_data:
            self.mapgen_cfg = parse_mapgen_config(self.config.mapgen_config_data)
        else:
            self.mapgen_cfg = load_mapgen_config(self.config.mapgen_config_path)
        if self.config.map_structures_data:
            self.structure_defs = parse_structures_config(self.config.map_structures_data)
        else:
            self.structure_defs = load_structures_config(self.config.map_structures_path)
        self.static_map_layout: StaticMapLayout | None = None
        if self.config.static_map_data:
            self.static_map_layout = parse_map_layout(self.config.static_map_data)
        elif str(self.config.static_map_path).strip():
            self.static_map_layout = load_map_layout(self.config.static_map_path)
        self._validate_static_map_references()
        if self.config.recipes_data:
            self.recipes = parse_recipes(self.config.recipes_data)
        else:
            self.recipes = load_recipes(self.config.recipes_path)
        if self.config.statuses_data:
            self.status_defs = parse_statuses(self.config.statuses_data)
        else:
            self.status_defs = load_statuses(self.config.statuses_path)
        if self.config.spells_data:
            self.spell_defs = parse_spells(self.config.spells_data)
        else:
            self.spell_defs = load_spells(self.config.spells_path)
        if self.config.enchantments_data:
            self.enchant_defs = parse_enchantments(self.config.enchantments_data)
        else:
            self.enchant_defs = load_enchantments(self.config.enchantments_path)
        self._recipe_ids = sorted(self.recipes.keys())
        self._validate_recipe_references()
        self._validate_effect_references()
        self._rng = random.Random(0)
        self.possible_agents = [f"agent_{i}" for i in range(self.config.n_agents)]
        self.agents = list(self.possible_agents)
        self.state: Optional[EnvState] = None
        self._last_info: Dict[str, Dict[str, object]] = {}
        self._render_window: Optional[RenderWindow] = None
        self._winner_announced: bool = False
        self._episode_metrics: Dict[str, Dict[str, object]] = {}
        self._walkable_tile_count: int = 1
        self._episode_combat_exchanges: int = 0
        self._episode_any_enemy_seen: bool = False
        self._episode_timeout_no_contact: bool = False
        self._episode_terminal_rewards_applied: bool = False
        self._next_faction_id: int = 1
        self._guard_assignments: Dict[str, str] = {}
        self._revived_this_step: set[str] = set()
        self._deferred_agent_rewards: Dict[str, float] = {}
        self._deferred_agent_events: Dict[str, List[str]] = {}
        self._join_bonus_awarded_invitees: set[str] = set()
        self._faction_action_next_step: Dict[str, Dict[str, int]] = {}
        self._team_pair_reward_counts: Dict[str, int] = {}
        self._team_pair_last_reward_step: Dict[str, int] = {}
        self._defending_agents: set[str] = set()

    def action_space(self, agent_id: str) -> Tuple[int, int]:
        if agent_id not in self.possible_agents:
            raise KeyError(f"Unknown agent: {agent_id}")
        return (0, int(ACTION_MAX))

    def observation_space(self, agent_id: str) -> Dict[str, object]:
        if agent_id not in self.possible_agents:
            raise KeyError(f"Unknown agent: {agent_id}")
        profile_name = self._resolve_profile_name(
            agent_id, self.possible_agents.index(agent_id)
        )
        profile = self._profile_by_name(profile_name)
        keys = ["step", "alive", "profile", "race", "class", "faction"]
        if profile.include_grid:
            keys.append("local_tiles")
        if profile.include_stats:
            keys.append("stats")
        if profile.include_inventory:
            keys.append("inventory")
        return {"type": "dict", "keys": keys}

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, object]] = None
    ):
        if seed is not None:
            self._rng.seed(seed)

        if self.static_map_layout is not None:
            grid = [list(row) for row in self.static_map_layout.grid]
            biome_map = dict(self.static_map_layout.biomes)
        else:
            grid, biome_map = self._generate_world_terrain()
        starts = sample_walkable_positions(
            grid, self.tiles, self.config.n_agents, self._rng
        )
        if self.config.combat_training_mode:
            starts = self._cluster_agent_starts_for_combat(grid=grid, starts=starts)

        agents: Dict[str, AgentState] = {}
        for i, agent_id in enumerate(self.possible_agents):
            pos = starts[i]
            profile_name = self._resolve_profile_name(agent_id, i)
            profile = self._profile_by_name(profile_name)
            race_name = self._resolve_race_name(agent_id, i)
            race = self._race_by_name(race_name)
            class_name = self._resolve_class_name(agent_id, i)
            cls = self._class_by_name(class_name)
            agent = AgentState(
                agent_id=agent_id,
                position=pos,
                profile_name=profile.name,
                race_name=race.name,
                class_name=cls.name,
                hp=profile.max_hp,
                max_hp=profile.max_hp,
                mana=max(0, race.intellect + 4),
                max_mana=max(0, race.intellect + 4),
                hunger=profile.max_hunger,
                max_hunger=profile.max_hunger,
                strength=race.strength,
                dexterity=race.dexterity,
                intellect=race.intellect,
            )
            agent.known_spells = self._default_known_spells_for_class(cls.name)
            self._apply_class_modifiers(agent, cls)
            if cls.starting_items:
                agent.inventory.extend(list(cls.starting_items))
            agent.visited.add(pos)
            agents[agent_id] = agent

        self.state = EnvState(
            grid=grid,
            tile_interactions={},
            tile_harvest_counts={},
            ground_items={},
            agents=agents,
            chests={},
            plant_plots={},
            agent_statuses={aid: [] for aid in self.possible_agents},
            item_metadata={},
            faction_leaders={},
            pending_faction_invites={},
            biomes=biome_map,
            step_count=0,
        )
        self._next_faction_id = 1
        self._join_bonus_awarded_invitees = set()
        self._deferred_agent_rewards = {}
        self._deferred_agent_events = {}
        self._faction_action_next_step = {}
        self._team_pair_reward_counts = {}
        self._team_pair_last_reward_step = {}
        self.state.chests = self._spawn_chests(starts)
        occupied = starts + list(self.state.chests.keys())
        self.state.resource_nodes = self._spawn_resource_nodes(occupied=occupied)
        occupied.extend(list(self.state.resource_nodes.keys()))
        self.state.stations = self._spawn_stations(occupied=occupied)
        occupied.extend(list(self.state.stations.keys()))
        self.state.animals = self._spawn_animals(occupied=occupied)
        occupied.extend(
            [a.position for a in self.state.animals.values() if a.alive]
        )
        self.state.monsters = self._spawn_monsters(
            occupied=occupied
        )
        self._walkable_tile_count = max(
            1,
            sum(
                1
                for row in self.state.grid
                for tile_id in row
                if self.tiles[tile_id].walkable
            ),
        )
        self._episode_combat_exchanges = 0
        self._episode_any_enemy_seen = False
        self._episode_timeout_no_contact = False
        self._episode_terminal_rewards_applied = False
        self._episode_metrics = {}
        for aid in self.possible_agents:
            visible = self._visible_tile_coords(aid)
            first_enemy_visible = self._enemy_visible(aid)
            self._episode_any_enemy_seen = self._episode_any_enemy_seen or first_enemy_visible
            self._episode_metrics[aid] = {
                "seen_tiles": set(visible),
                "steps_since_new_tile": 0,
                "first_enemy_seen_step": 0 if first_enemy_visible else None,
                "enemy_visible_steps": 1 if first_enemy_visible else 0,
                "last_enemy_distance": self._nearest_opponent_distance(aid),
                "enemy_distance_delta_sum": 0.0,
                "enemy_distance_delta_count": 0,
                "combat_exchanges": 0,
                "ever_enemy_seen": bool(first_enemy_visible),
            }
        self._winner_announced = False
        self.agents = list(self.possible_agents)
        obs = {aid: self._build_observation(aid) for aid in self.possible_agents}
        info = {
            aid: {
                "action_mask": [1] * (int(ACTION_MAX) + 1),
                "alive": True,
                "profile": self.state.agents[aid].profile_name,
                "race": self.state.agents[aid].race_name,
                "class": self.state.agents[aid].class_name,
                "faction_id": int(self.state.agents[aid].faction_id),
            }
            for aid in self.possible_agents
        }
        self._last_info = info
        if self.config.render_enabled and self._render_window is not None:
            self._render_window.update_state(
                self.state, focus_choices=self.possible_agents
            )
        return obs, info

    def step(self, actions: Dict[str, int]):
        if self.state is None:
            raise RuntimeError("Environment must be reset before step")
        self._guard_assignments = {}
        self._defending_agents = set()
        self._revived_this_step = set()
        self._deferred_agent_rewards = {}
        self._deferred_agent_events = {}
        self._prune_expired_invites()

        rewards = {aid: 0.0 for aid in self.possible_agents}
        terminations = {aid: False for aid in self.possible_agents}
        truncations = {aid: False for aid in self.possible_agents}
        info = {aid: {"events": []} for aid in self.possible_agents}
        reward_components = {
            aid: {
                "action_total": 0.0,
                "survival": 0.0,
                "search_explore": 0.0,
                "profile_shape": 0.0,
                "teamwork": 0.0,
                "treasure": 0.0,
                "focus": 0.0,
                "terminal": 0.0,
            }
            for aid in self.possible_agents
        }
        pre_enemy_distance = {
            aid: self._nearest_opponent_distance(aid) for aid in self.possible_agents
        }
        pre_enemy_visible = {
            aid: self._enemy_visible(aid) for aid in self.possible_agents
        }

        for aid in self.possible_agents:
            agent = self.state.agents[aid]
            if agent.hp <= 0 or not agent.alive:
                if agent.alive and agent.hp <= 0:
                    rewards[aid] -= 1.0
                    info[aid]["events"].append("death")
                agent.alive = False
                terminations[aid] = True
                continue

            action = int(actions.get(aid, ACTION_WAIT))
            action = self._status_adjust_action(aid=aid, action=action, info=info)
            info[aid]["action"] = action
            delta_reward, events = self._apply_action(aid, action)
            rewards[aid] += delta_reward
            reward_components[aid]["action_total"] += float(delta_reward)
            info[aid]["events"].extend(events)
            survival_delta = self._apply_survival_costs(agent, rewards, aid, info)
            reward_components[aid]["survival"] += float(survival_delta)
            status_delta = self._tick_agent_statuses(aid=aid, rewards=rewards, info=info)
            reward_components[aid]["survival"] += float(status_delta)
            search_delta = self._apply_search_and_exploration_rewards(
                aid=aid,
                rewards=rewards,
                info=info,
                pre_enemy_distance=pre_enemy_distance[aid],
                pre_enemy_visible=pre_enemy_visible[aid],
            )
            reward_components[aid]["search_explore"] += float(search_delta)

            if agent.hp <= 0:
                agent.alive = False
                terminations[aid] = True
                rewards[aid] -= 1.0
                reward_components[aid]["terminal"] -= 1.0
                info[aid]["events"].append("death")

        for revived_aid in sorted(self._revived_this_step):
            if revived_aid not in self.state.agents:
                continue
            revived = self.state.agents[revived_aid]
            if revived.alive and revived.hp > 0:
                terminations[revived_aid] = False
                info[revived_aid]["events"].append("revived")

        for aid, delta in self._deferred_agent_rewards.items():
            if aid not in rewards:
                continue
            rewards[aid] += float(delta)
            reward_components[aid]["teamwork"] += float(delta)
        for aid, events in self._deferred_agent_events.items():
            if aid not in info:
                continue
            info[aid]["events"].extend(list(events))

        self._apply_monster_turn(rewards, terminations, info)
        self._apply_animal_turn(info=info)
        self._tick_fires()
        self.state.step_count += 1

        if self.state.step_count >= self.config.max_steps:
            for aid in self.possible_agents:
                truncations[aid] = True

        for aid in self.possible_agents:
            profile = self._profile_for_agent(aid)
            prof_delta = profile.reward_adjustment(
                events=info[aid]["events"],
                died=terminations[aid],
            )
            rewards[aid] += prof_delta
            reward_components[aid]["profile_shape"] += float(prof_delta)
            agent = self.state.agents[aid]
            info[aid]["alive"] = agent.alive
            info[aid]["profile"] = agent.profile_name
            info[aid]["race"] = agent.race_name
            info[aid]["class"] = agent.class_name
            info[aid]["faction_id"] = int(agent.faction_id)
            info[aid]["is_faction_leader"] = bool(self._is_faction_leader(aid))
            info[aid]["pending_invite_from_faction"] = self._pending_invite_faction_id(aid)
            info[aid]["teammate_distance"] = self._nearest_teammate_distance(aid)
            metrics = self._episode_metrics.get(aid, {})
            seen_tiles = int(len(metrics.get("seen_tiles", set())))
            info[aid]["new_tiles_seen_total"] = seen_tiles
            info[aid]["explore_coverage"] = float(seen_tiles) / float(self._walkable_tile_count)
            info[aid]["steps_since_new_tile"] = int(metrics.get("steps_since_new_tile", 0))
            info[aid]["first_enemy_seen_step"] = metrics.get("first_enemy_seen_step")
            info[aid]["enemy_visible_steps"] = int(metrics.get("enemy_visible_steps", 0))
            info[aid]["enemy_distance"] = self._nearest_opponent_distance(aid)
            dcnt = int(metrics.get("enemy_distance_delta_count", 0))
            dsum = float(metrics.get("enemy_distance_delta_sum", 0.0))
            info[aid]["enemy_distance_delta_mean"] = (dsum / dcnt) if dcnt > 0 else 0.0
            info[aid]["combat_exchanges"] = int(metrics.get("combat_exchanges", 0))
            info[aid]["ever_enemy_seen"] = bool(metrics.get("ever_enemy_seen", False))
            info[aid]["timeout_no_contact"] = bool(self._episode_timeout_no_contact)

        self._apply_team_rewards(
            rewards=rewards,
            reward_components=reward_components,
            info=info,
        )
        self._apply_treasure_rewards(
            rewards=rewards,
            reward_components=reward_components,
            info=info,
        )
        self._apply_focus_rewards(
            rewards=rewards,
            reward_components=reward_components,
            info=info,
        )

        alive_now = [
            aid
            for aid in self.possible_agents
            if self.state.agents[aid].alive and not truncations[aid]
        ]
        if (
            not self._winner_announced
            and self.config.n_agents > 1
            and len(alive_now) == 1
        ):
            winner = alive_now[0]
            info[winner]["events"].append(f"winner:{winner}")
            info[winner]["events"].append("episode_end:last_survivor")
            # End competitive episodes immediately once a single survivor remains.
            truncations[winner] = True
            self._winner_announced = True
        elif (
            not self._winner_announced
            and self.config.n_agents > 1
            and len(alive_now) == 0
        ):
            for aid in self.possible_agents:
                info[aid]["events"].append("winner:none")
            self._winner_announced = True
        elif (
            not self._winner_announced
            and self.config.n_agents > 1
            and self.state.step_count >= self.config.max_steps
        ):
            for aid in self.possible_agents:
                info[aid]["events"].append("winner:none")
            self._winner_announced = True
            self._episode_timeout_no_contact = (
                self._episode_combat_exchanges <= 0
                and not self._episode_any_enemy_seen
            )
            if self._episode_timeout_no_contact:
                for aid in self.possible_agents:
                    info[aid]["events"].append("episode_timeout_no_contact")

        episode_done = all(
            bool(terminations.get(aid, False) or truncations.get(aid, False))
            for aid in self.possible_agents
        )
        if episode_done and not self._episode_terminal_rewards_applied:
            self._apply_treasure_end_bonus(
                rewards=rewards,
                reward_components=reward_components,
                info=info,
            )
            if self._episode_combat_exchanges > 0:
                for aid in self.possible_agents:
                    term_delta = float(self.config.engagement_bonus)
                    rewards[aid] += term_delta
                    reward_components[aid]["terminal"] += term_delta
                    info[aid]["events"].append("episode_engagement_bonus")
            else:
                for aid in self.possible_agents:
                    info[aid]["events"].append("episode_no_combat")
            if self.state.step_count >= self.config.max_steps:
                for aid in self.possible_agents:
                    term_delta = -float(self.config.timeout_tie_penalty)
                    rewards[aid] += term_delta
                    reward_components[aid]["terminal"] += term_delta
                    info[aid]["events"].append("episode_timeout_tie")
                    info[aid]["timeout_no_contact"] = bool(self._episode_timeout_no_contact)
            self._episode_terminal_rewards_applied = True

        for aid in self.possible_agents:
            info[aid]["reward_components"] = {
                k: float(v) for k, v in reward_components[aid].items()
            }

        self.agents = [
            aid
            for aid in self.possible_agents
            if not terminations[aid] and not truncations[aid]
        ]

        observations = {
            aid: self._build_observation(aid)
            for aid in self.possible_agents
            if self.state.agents[aid].alive or truncations[aid]
        }
        self._last_info = info
        if self.config.render_enabled and self._render_window is not None:
            self._render_window.update_state(
                self.state, focus_choices=self.possible_agents
            )
        return observations, rewards, terminations, truncations, info

    def render(self, focus_agent: Optional[str] = None, zoom: int = 0) -> None:
        """Render only to the dedicated window (no CLI output mode)."""
        if not self.config.render_enabled or self.state is None:
            return
        if self._render_window is None:
            self.open_render_window()
        assert self._render_window is not None
        self._render_window.focus_var.set(focus_agent or "all")
        self._render_window.zoom_var.set(max(0, min(10, int(zoom))))
        self._render_window.update_state(self.state, focus_choices=self.possible_agents)

    def open_render_window(self, title: str = "RLRLGym Viewer") -> None:
        if not self.config.render_enabled:
            return
        if self._render_window is None:
            self._render_window = RenderWindow(self.tiles, title=title)
        if self.state is not None:
            self._render_window.update_state(
                self.state, focus_choices=self.possible_agents
            )

    def close_render_window(self) -> None:
        if self._render_window is not None:
            self._render_window.close()
            self._render_window = None

    def play_frames_in_window(
        self,
        states: List[EnvState],
        title: str = "RLRLGym Playback",
        playback_actions: Optional[List[Dict[str, object]]] = None,
        on_prev_episode: Optional[Callable[[], None]] = None,
        on_next_episode: Optional[Callable[[], None]] = None,
    ) -> None:
        if not self.config.render_enabled:
            return
        if self._render_window is None:
            self._render_window = RenderWindow(self.tiles, title=title)
        self._render_window.root.title(title)
        self._render_window.set_episode_navigation(
            on_prev_episode=on_prev_episode,
            on_next_episode=on_next_episode,
        )
        self._render_window.set_playback_states(
            states,
            focus_choices=self.possible_agents,
            action_log=playback_actions,
        )
        self._render_window.play()

    def capture_playback_state(self) -> EnvState:
        if self.state is None:
            raise RuntimeError(
                "Environment must be reset before capturing playback state"
            )
        return copy.deepcopy(self.state)

    def run_render_window(self) -> None:
        if self._render_window is None:
            self.open_render_window()
        assert self._render_window is not None
        self._render_window.run()

    def snapshot(self) -> Dict[str, object]:
        if self.state is None:
            raise RuntimeError("Cannot snapshot before reset")
        return copy.deepcopy(
            {
                "config": copy.deepcopy(self.config.__dict__),
                "state": self.state,
                "rng_state": self._rng.getstate(),
                "last_info": self._last_info,
            }
        )

    def load_snapshot(self, snap: Dict[str, object]) -> None:
        self.config = EnvConfig(**snap["config"])
        self.state = snap["state"]
        self._rng.setstate(snap["rng_state"])
        self._last_info = snap.get("last_info", {})

    def _apply_action(self, aid: str, action: int) -> Tuple[float, List[str]]:
        assert self.state is not None
        agent = self.state.agents[aid]
        reward = 0.0
        events: List[str] = []
        adjacent_hostile = self._has_adjacent_hostile(agent=agent, actor_id=aid)
        if (
            self.config.combat_training_mode
            and adjacent_hostile
            and action != ACTION_ATTACK
        ):
            reward -= float(self.config.missed_attack_opportunity_penalty)
            events.append("missed_attack_opportunity")

        if action in MOVE_DELTAS:
            old_food_distance = self._nearest_food_distance(agent.position)
            athletics_level = self._skill_level(agent, "athletics")
            encumbrance_penalty = self._encumbrance_penalty(agent, athletics_level)
            enc_ratio = self._encumbrance_ratio(agent)
            if enc_ratio > 1.8:
                reward -= 0.03
                events.append("dragging:immobile")
                return reward, events
            dr, dc = MOVE_DELTAS[action]
            nr, nc = agent.position[0] + dr, agent.position[1] + dc
            if self._walkable(nr, nc):
                old_pos = agent.position
                agent.position = (nr, nc)
                events.append(f"move:{old_pos}->{agent.position}")
                tile_here = self.state.grid[nr][nc]
                if tile_here == "spike_trap":
                    trap_damage = 2
                    agent.hp = max(0, int(agent.hp) - trap_damage)
                    events.append(f"trap_hit:spike:{trap_damage}")
                    reward -= 0.03 * float(trap_damage)
                self._gain_skill_xp(agent, "athletics", 1, events)
                self._gain_skill_xp(agent, "exploration", 1, events)
                reward += MOVE_VALID_REWARD
                reward += float(self.config.move_bias_reward)
                reward -= MOVE_STEP_COST
                reward -= encumbrance_penalty
                new_food_distance = self._nearest_food_distance(agent.position)
                if (
                    old_food_distance is not None
                    and new_food_distance is not None
                    and new_food_distance < old_food_distance
                ):
                    reward += MOVE_FOOD_PROGRESS_REWARD
                    events.append("food_progress")
                elif (
                    old_food_distance is not None
                    and new_food_distance is not None
                    and new_food_distance > old_food_distance
                ):
                    reward -= MOVE_FOOD_REGRESS_PENALTY
                    events.append("food_regress")
                if agent.position not in agent.visited:
                    reward += 0.05
                    events.append("explore")
                    agent.visited.add(agent.position)
                # Penalize immediate backtracking loops.
                if (
                    len(agent.recent_positions) >= 2
                    and agent.recent_positions[-2] == agent.position
                ):
                    reward -= max(0.0, 0.02 - 0.002 * athletics_level)
                    events.append("stutter_penalty")
                repeat_window = max(2, int(self.config.repeat_visit_window))
                if agent.recent_positions[-repeat_window:].count(agent.position) >= 2:
                    reward -= float(self.config.repeat_visit_penalty)
                    events.append("repeat_visit_penalty")
                agent.recent_positions.append(agent.position)
                agent.recent_positions = agent.recent_positions[-max(5, repeat_window):]
                agent.wait_streak = 0
            else:
                reward -= max(0.0, 0.02 - 0.002 * athletics_level)
                events.append("bump")

        elif action == ACTION_WAIT:
            agent.wait_streak += 1
            reward -= 0.01
            events.append("wait")
            if (
                not self._enemy_visible(aid)
                and (agent.hunger / max(1, agent.max_hunger))
                >= float(self.config.wait_safe_hunger_ratio)
            ):
                reward -= float(self.config.wait_no_enemy_penalty)
                events.append("wait_no_enemy_penalty")
            if agent.wait_streak > 3:
                reward -= 0.02
                events.append("wait_loop_penalty")

        elif action in (ACTION_LOOT, ACTION_PICKUP):
            reward += self._pickup_from_tile(agent, events)

        elif action == ACTION_EAT:
            pre_hunger = agent.hunger
            if "ration" in agent.inventory:
                agent.inventory.remove("ration")
                agent.hunger = min(agent.max_hunger, agent.hunger + 8)
                hunger_gain = agent.hunger - pre_hunger
                reward += EAT_PER_HUNGER_GAIN_REWARD * hunger_gain
                events.append("eat:ration")
            elif "fruit" in agent.inventory:
                agent.inventory.remove("fruit")
                agent.hunger = min(agent.max_hunger, agent.hunger + 4)
                hunger_gain = agent.hunger - pre_hunger
                reward += EAT_PER_HUNGER_GAIN_REWARD * hunger_gain
                events.append("eat:fruit")
            else:
                reward -= 0.02
                events.append("eat_fail")
                hunger_gain = 0

            if (
                hunger_gain > 0
                and pre_hunger >= int(agent.max_hunger * EAT_WASTE_THRESHOLD)
            ):
                reward -= EAT_WASTE_PENALTY
                events.append("eat_waste_penalty")

        elif action == ACTION_EQUIP:
            if agent.inventory:
                item = agent.inventory.pop(0)
                base_item = self._item_base_id(item)
                if int(self.base_durability_by_item.get(base_item, 0)) > 0 and item not in self.state.item_metadata:
                    crafting_skill = self._skill_level(agent, "crafting")
                    token = self._make_item_instance(
                        base_id=base_item,
                        kind="durable_item",
                        metadata={
                            "durability": int(self.base_durability_by_item.get(base_item, 0)) + max(0, crafting_skill // 2),
                            "max_durability": int(self.base_durability_by_item.get(base_item, 0)) + max(0, crafting_skill // 2),
                            "crafted_by_skill": int(crafting_skill),
                        },
                    )
                    item = token
                    base_item = self._item_base_id(item)
                armor_slot = self.armor_slot_by_item.get(base_item)
                if armor_slot is not None:
                    target_slot = armor_slot
                    if armor_slot == RING_ITEM_SLOT:
                        target_slot = RING_ARMOR_SLOTS[0]
                        for candidate in RING_ARMOR_SLOTS:
                            if agent.armor_slots.get(candidate) is None:
                                target_slot = candidate
                                break
                    replaced_item = agent.armor_slots.get(target_slot)
                    if replaced_item:
                        for idx in range(len(agent.equipped) - 1, -1, -1):
                            if agent.equipped[idx] == replaced_item:
                                agent.equipped.pop(idx)
                                break
                        agent.inventory.append(replaced_item)
                        events.append(f"unequip:{target_slot}:{replaced_item}")
                    agent.armor_slots[target_slot] = item
                    armor_slot = target_slot
                agent.equipped.append(item)
                reward += 0.08
                events.append(f"equip:{item}")
                if armor_slot is not None:
                    events.append(f"equip_slot:{armor_slot}")
            else:
                reward -= 0.01

        elif action == ACTION_USE:
            reward += self._use_item_or_spell(actor=agent, aid=aid, events=events)

        elif action == ACTION_GIVE:
            reward += self._give_to_ally(actor=agent, actor_id=aid, events=events)

        elif action == ACTION_TRADE:
            reward += self._trade_with_ally(actor=agent, actor_id=aid, events=events)

        elif action == ACTION_REVIVE:
            reward += self._revive_ally(actor=agent, actor_id=aid, events=events)

        elif action == ACTION_GUARD:
            reward += self._guard_ally(actor=agent, actor_id=aid, events=events)

        elif action == ACTION_DEFEND:
            reward += self._defend(actor=agent, actor_id=aid, events=events)

        elif action == ACTION_LEAVE_FACTION:
            reward += self._leave_faction(actor=agent, actor_id=aid, events=events)

        elif action == ACTION_ACCEPT_INVITE:
            reward += self._accept_faction_invite(
                actor=agent, actor_id=aid, events=events
            )

        elif action == ACTION_INTERACT:
            reward += self._interact(agent, aid, events)

        elif action == ACTION_INTERACT_RESOURCE:
            specific_reward, handled = self._interact_resource_node(
                actor=agent, actor_id=aid, events=events
            )
            reward += specific_reward if handled else -0.01

        elif action == ACTION_INTERACT_STATION:
            specific_reward, handled = self._interact_station(
                actor=agent, actor_id=aid, events=events
            )
            reward += specific_reward if handled else -0.01

        elif action == ACTION_INTERACT_SHEAR:
            specific_reward, handled = self._interact_shear(
                actor=agent, actor_id=aid, events=events
            )
            reward += specific_reward if handled else -0.01

        elif action == ACTION_INTERACT_HARVEST:
            specific_reward, handled = self._interact_harvest_plant(
                actor=agent, actor_id=aid, events=events
            )
            reward += specific_reward if handled else -0.01

        elif action == ACTION_INTERACT_MINE:
            specific_reward, handled = self._interact_mine_tile(
                actor=agent, actor_id=aid, events=events
            )
            reward += specific_reward if handled else -0.01

        elif action == ACTION_INTERACT_FORAGE:
            specific_reward, handled = self._interact_forage_tile(
                actor=agent, actor_id=aid, events=events
            )
            reward += specific_reward if handled else -0.01

        elif action == ACTION_INTERACT_PLANT:
            specific_reward, handled = self._interact_plant(
                actor=agent, actor_id=aid, events=events
            )
            reward += specific_reward if handled else -0.01

        elif action == ACTION_INTERACT_FACTION:
            specific_reward, handled = self._handle_faction_interact(
                actor=agent, actor_id=aid, events=events
            )
            reward += specific_reward if handled else -0.01

        elif action == ACTION_INTERACT_CHEST:
            reward += self._interact_chest(agent=agent, events=events)

        elif action == ACTION_INTERACT_SHRINE:
            reward += self._interact_shrine(actor=agent, events=events)

        elif action == ACTION_INTERACT_DRINK:
            reward += self._interact_drink(actor=agent, events=events)

        elif action == ACTION_ATTACK:
            if self.config.combat_training_mode and adjacent_hostile:
                reward += 0.02
                events.append("combat_engage_bonus")
            reward += self._attack(agent, aid, events)

        return reward, events

    def _interact(self, actor: AgentState, actor_id: str, events: List[str]) -> float:
        assert self.state is not None
        reward = 0.0
        prefer_faction = self._pending_invite_faction_id(actor_id) is not None
        if not prefer_faction:
            for other_id, other in self.state.agents.items():
                if other_id == actor_id or not other.alive:
                    continue
                if self._manhattan(actor.position, other.position) == 1:
                    prefer_faction = True
                    break
        if prefer_faction:
            faction_reward, faction_handled = self._handle_faction_interact(
                actor=actor, actor_id=actor_id, events=events
            )
            if faction_handled:
                return reward + faction_reward
        resource_reward, resource_handled = self._interact_resource_node(
            actor=actor, actor_id=actor_id, events=events
        )
        if resource_handled:
            return reward + resource_reward
        station_reward, station_handled = self._interact_station(
            actor=actor, actor_id=actor_id, events=events
        )
        if station_handled:
            return reward + station_reward
        fire_reward, fire_handled = self._interact_fire(
            actor=actor, actor_id=actor_id, events=events
        )
        if fire_handled:
            return reward + fire_reward
        shear_reward, shear_handled = self._interact_shear(actor=actor, actor_id=actor_id, events=events)
        if shear_handled:
            return reward + shear_reward
        harvest_reward, harvest_handled = self._interact_harvest_plant(
            actor=actor, actor_id=actor_id, events=events
        )
        if harvest_handled:
            return reward + harvest_reward
        mine_reward, mine_handled = self._interact_mine_tile(
            actor=actor, actor_id=actor_id, events=events
        )
        if mine_handled:
            return reward + mine_reward
        forage_reward, forage_handled = self._interact_forage_tile(
            actor=actor, actor_id=actor_id, events=events
        )
        if forage_handled:
            return reward + forage_reward
        plant_reward, plant_handled = self._interact_plant(
            actor=actor, actor_id=actor_id, events=events
        )
        if plant_handled:
            return reward + plant_reward
        faction_reward, faction_handled = self._handle_faction_interact(
            actor=actor, actor_id=actor_id, events=events
        )
        if faction_handled:
            return reward + faction_reward
        chest_reward = self._interact_chest(agent=actor, events=events)
        if chest_reward > -0.01:
            return reward + chest_reward
        if self.state.grid[actor.position[0]][actor.position[1]] == "shrine":
            return reward + self._interact_shrine(actor=actor, events=events)
        if self.state.grid[actor.position[0]][actor.position[1]] in WATER_TILE_IDS:
            return reward + self._interact_drink(actor=actor, events=events)
        return reward + self._interact_generic_tile(actor, actor_id, events)

    def _interact_generic_tile(self, actor: AgentState, actor_id: str, events: List[str]) -> float:
        assert self.state is not None
        reward = 0.0
        r, c = actor.position

        tile_id = self.state.grid[r][c]
        tile = self.tiles[tile_id]

        n_interactions = self.state.tile_interactions.get((r, c), 0)
        if tile.max_interactions > 0 and n_interactions < tile.max_interactions:
            self.state.tile_interactions[(r, c)] = n_interactions + 1
            if tile_id == "shrine":
                actor.hp = min(actor.max_hp, actor.hp + 1)
                reward += 0.1
                events.append("interact:shrine")
            elif tile_id in WATER_TILE_IDS:
                actor.hunger = min(actor.max_hunger, actor.hunger + 1)
                reward += 0.04
                events.append("interact:water")
            else:
                reward += 0.05
                events.append(f"interact:{tile_id}")
        else:
            events.append("interact_exhausted")
            reward -= 0.02
        return reward

    def _interact_chest(self, agent: AgentState, events: List[str]) -> float:
        assert self.state is not None
        chest = self.state.chests.get(agent.position)
        if chest and not chest.opened:
            events.append("interact:chest")
            return self._pickup_from_tile(agent, events)
        events.append("interact_chest_fail")
        return -0.01

    def _interact_shrine(self, actor: AgentState, events: List[str]) -> float:
        assert self.state is not None
        r, c = actor.position
        if self.state.grid[r][c] != "shrine":
            events.append("interact_shrine_fail")
            return -0.01
        tile = self.tiles.get("shrine")
        if tile is None:
            events.append("interact_shrine_fail")
            return -0.01
        used = int(self.state.tile_interactions.get((r, c), 0))
        if used >= max(1, int(tile.max_interactions)):
            events.append("interact_exhausted")
            return -0.02
        self.state.tile_interactions[(r, c)] = used + 1
        actor.hp = min(actor.max_hp, actor.hp + 1)
        events.append("interact:shrine")
        return 0.1

    def _interact_drink(self, actor: AgentState, events: List[str]) -> float:
        assert self.state is not None
        r, c = actor.position
        tile_id = self.state.grid[r][c]
        if tile_id not in WATER_TILE_IDS:
            events.append("interact_drink_fail")
            return -0.01
        tile = self.tiles.get(tile_id)
        if tile is None:
            events.append("interact_drink_fail")
            return -0.01
        used = int(self.state.tile_interactions.get((r, c), 0))
        if used >= max(1, int(tile.max_interactions)):
            events.append("interact_exhausted")
            return -0.02
        self.state.tile_interactions[(r, c)] = used + 1
        actor.hunger = min(actor.max_hunger, actor.hunger + 1)
        events.append("interact:water")
        return 0.04

    def _interact_fire(
        self, actor: AgentState, actor_id: str, events: List[str]
    ) -> Tuple[float, bool]:
        assert self.state is not None
        r, c = actor.position
        tile_id = self.state.grid[r][c]
        if tile_id != "campfire":
            return 0.0, False
        for fuel_item, fuel_gain in (
            ("log", FIRE_FUEL_PER_LOG),
            ("wood", FIRE_FUEL_PER_WOOD),
            ("stick", FIRE_FUEL_PER_STICK),
        ):
            taken = self._pop_first_base_item(actor.inventory, fuel_item)
            if taken is None:
                continue
            fuel = int(self.state.tile_interactions.get((r, c), 0))
            fuel = min(FIRE_FUEL_MAX, fuel + int(fuel_gain))
            self.state.tile_interactions[(r, c)] = fuel
            events.append(f"fire_refuel:{fuel_item}:{fuel}")
            return 0.08, True
        events.append("fire_refuel_fail:no_fuel")
        return -0.01, True

    def _is_unbreakable_edge(self, pos: Tuple[int, int]) -> bool:
        assert self.state is not None
        r, c = pos
        return (
            r <= 0
            or c <= 0
            or r >= len(self.state.grid) - 1
            or c >= len(self.state.grid[0]) - 1
        )

    def _interact_mine_tile(
        self, actor: AgentState, actor_id: str, events: List[str]
    ) -> Tuple[float, bool]:
        assert self.state is not None
        r, c = actor.position
        mining = self._skill_level(actor, "mining")
        mining_tool_bonus = self._max_equipped_tool_bonus(actor, skill="mining")

        # Harvest nearby trees for sticks; axes can also fell trees into logs.
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nr < 0 or nc < 0 or nr >= len(self.state.grid) or nc >= len(self.state.grid[0]):
                continue
            adj_tile = self.state.grid[nr][nc]
            if adj_tile not in TREE_TILE_IDS:
                continue
            if self._harvest_exhausted((nr, nc), adj_tile, events):
                return -0.02, True
            woodcutting = self._skill_level(actor, "woodcutting")
            axe_bonus = max(
                self._max_equipped_tool_bonus(actor, skill="woodcutting", category="axe"),
                self._max_equipped_tool_bonus(actor, skill="woodcutting", category="handaxe"),
            )
            if axe_bonus <= 0:
                continue
            used_tool = self._equipped_tool_for_category(actor, "axe") or self._equipped_tool_for_category(actor, "handaxe")
            stick_qty = 1 + (woodcutting // 6) + (axe_bonus // 3)
            stick_added = self._add_item_or_drop(actor, "stick", max(1, stick_qty), events)
            self._record_harvest((nr, nc))
            self._gain_skill_xp(actor, "woodcutting", max(1, 2 + stick_qty), events)
            events.append(f"harvest_tree:stick:{stick_qty}:{stick_added}:{nr}:{nc}")
            reward = 0.05 + (0.01 * float(max(1, stick_qty)))
            progress = 1 + axe_bonus + max(0, woodcutting // 5)
            used = int(self.state.tile_interactions.get((nr, nc), 0)) + progress
            self.state.tile_interactions[(nr, nc)] = used
            events.append(f"chop_tree_progress:{nr}:{nc}:{used}/{TREE_CHOP_PROGRESS_REQUIRED}")
            if used >= TREE_CHOP_PROGRESS_REQUIRED:
                floor_id = (
                    self.mapgen_cfg.floor_fallback_id
                    if self.mapgen_cfg.floor_fallback_id in self.tiles
                    else "floor"
                )
                self.state.grid[nr][nc] = floor_id
                self.state.tile_interactions.pop((nr, nc), None)
                log_qty = 1 + max(0, axe_bonus // 2)
                log_added = self._add_item_or_drop(actor, "log", log_qty, events)
                events.append(f"chop_tree_felled:{nr}:{nc}:log:{log_qty}:{log_added}")
                if self._rng.random() < TREE_SAPLING_DROP_CHANCE:
                    sapling_added = self._add_item_or_drop(actor, "sapling", 1, events)
                    events.append(f"chop_tree_sapling:{nr}:{nc}:{sapling_added}")
                reward += 0.08 + (0.02 * float(log_qty))
            if used_tool is not None:
                self._consume_item_durability(actor, used_tool, 1, events)
            if stick_added <= 0:
                reward -= 0.02
                events.append("carry_blocked")
            return reward, True

        # Mine adjacent stone walls; interior walls break into stone floor after 5 mines.
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nr < 0 or nc < 0 or nr >= len(self.state.grid) or nc >= len(self.state.grid[0]):
                continue
            adj_tile = self.state.grid[nr][nc]
            if adj_tile not in MINABLE_WALL_TILE_IDS:
                continue
            if self._harvest_exhausted((nr, nc), adj_tile, events):
                return -0.02, True
            if self._is_unbreakable_edge((nr, nc)):
                events.append("mine_wall_unbreakable_edge")
                return -0.01, True
            used = int(self.state.tile_interactions.get((nr, nc), 0)) + 1
            self.state.tile_interactions[(nr, nc)] = used
            qty = 1 + (mining // 3) + (mining_tool_bonus // 2)
            added = self._add_item_or_drop(actor, "stone", max(1, qty), events)
            self._record_harvest((nr, nc))
            self._gain_skill_xp(actor, "mining", max(1, qty * 2), events)
            used_pickaxe = self._equipped_tool_for_category(actor, "pickaxe")
            if used_pickaxe is not None:
                self._consume_item_durability(actor, used_pickaxe, 1, events)
            if used >= 5:
                self.state.grid[nr][nc] = "stone_floor" if "stone_floor" in self.tiles else (
                    self.mapgen_cfg.floor_fallback_id if self.mapgen_cfg.floor_fallback_id in self.tiles else "floor"
                )
                self.state.tile_interactions.pop((nr, nc), None)
                events.append(f"mine_wall_broken:{nr}:{nc}")
            else:
                events.append(f"mine_wall:{nr}:{nc}:used={used}/5")
            reward = 0.06 + (0.01 * float(max(1, qty)))
            if added <= 0:
                reward -= 0.02
                events.append("carry_blocked")
            return reward, True

        # Mine stone floor underfoot. It depletes after 5 interactions but remains stone.
        tile_id = self.state.grid[r][c]
        if tile_id in MINABLE_GROUND_TILE_IDS:
            if self._harvest_exhausted((r, c), tile_id, events):
                return -0.02, True
            used = int(self.state.tile_interactions.get((r, c), 0))
            max_uses = 5
            if used >= max_uses:
                events.append("mine_ground_depleted")
                return -0.01, True
            used += 1
            self.state.tile_interactions[(r, c)] = used
            qty = 1 + (mining // 4) + (mining_tool_bonus // 2)
            added = self._add_item_or_drop(actor, "stone", max(1, qty), events)
            self._record_harvest((r, c))
            flint_added = 0
            flint_chance = min(0.35, STONE_FLOOR_FLINT_CHANCE + (0.01 * float(mining_tool_bonus)))
            if self._rng.random() < flint_chance:
                flint_added = self._add_item_or_drop(actor, "flint", 1, events)
            self._gain_skill_xp(actor, "mining", max(1, qty * 2), events)
            used_pickaxe = self._equipped_tool_for_category(actor, "pickaxe")
            if used_pickaxe is not None:
                self._consume_item_durability(actor, used_pickaxe, 1, events)
            events.append(
                f"mine_ground:{tile_id}:stone={qty}:{added}:flint={flint_added}:used={used}/5"
            )
            reward = 0.05 + (0.01 * float(max(1, qty)))
            if (added + flint_added) <= 0:
                reward -= 0.02
                events.append("carry_blocked")
            return reward, True
        return 0.0, False

    def _interact_forage_tile(
        self, actor: AgentState, actor_id: str, events: List[str]
    ) -> Tuple[float, bool]:
        assert self.state is not None
        pos = actor.position
        tile_id = self.state.grid[pos[0]][pos[1]]
        forage_tiles = {"grass", "bush"}
        if tile_id not in forage_tiles:
            return 0.0, False
        tile = self.tiles.get(tile_id)
        if tile is None or not tile.loot_table:
            return 0.0, False
        if self._harvest_exhausted(pos, tile_id, events):
            return -0.02, True
        used = int(self.state.tile_interactions.get(pos, 0))
        max_uses = max(1, int(tile.max_interactions))
        if used >= max_uses:
            events.append(f"forage_depleted:{tile_id}")
            return -0.01, True
        self.state.tile_interactions[pos] = used + 1
        self._record_harvest(pos)
        foraging = self._skill_level(actor, "foraging")
        qty = 1 + (foraging // 5)
        added_total = 0
        for _ in range(max(1, qty)):
            item = str(self._rng.choice(tile.loot_table))
            added_total += int(self._add_item_or_drop(actor, item, 1, events))
            events.append(f"forage:{tile_id}:{item}")
        self._gain_skill_xp(actor, "foraging", max(1, 2 + qty), events)
        reward = 0.05 + (0.01 * float(max(1, qty)))
        if added_total <= 0:
            reward -= 0.02
            events.append("carry_blocked")
        return reward, True

    def _interact_shear(
        self, actor: AgentState, actor_id: str, events: List[str]
    ) -> Tuple[float, bool]:
        assert self.state is not None
        for animal in self.state.animals.values():
            if not animal.alive:
                continue
            if animal.position != actor.position:
                continue
            if not animal.can_shear or not animal.shear_item:
                continue
            if animal.age < animal.mature_age:
                events.append(f"shear_fail:young:{animal.animal_id}")
                return -0.01, True
            if animal.sheared and animal.wool_regrow > 0:
                events.append(f"shear_fail:regrow:{animal.animal_id}")
                return -0.01, True
            added = self._add_item_or_drop(actor, animal.shear_item, 1, events)
            animal.sheared = True
            animal.wool_regrow = max(1, int(animal.shear_regrow_max) or 6)
            events.append(f"shear:{animal.animal_id}:{animal.shear_item}:{added}")
            return 0.1, True
        return 0.0, False

    def _interact_plant(
        self, actor: AgentState, actor_id: str, events: List[str]
    ) -> Tuple[float, bool]:
        assert self.state is not None
        r, c = actor.position
        tile_id = self.state.grid[r][c]
        if tile_id not in {"floor", "stone_floor", "grass", "bush"}:
            return 0.0, False
        if (r, c) in self.state.plant_plots:
            return 0.0, False
        crop_id = self._first_plantable_crop(actor.inventory)
        if not crop_id:
            return 0.0, False
        crop = PLANT_TYPES[crop_id]
        seed_item = str(crop["seed_item"])
        if self._pop_first_base_item(actor.inventory, seed_item) is None:
            return 0.0, False
        self.state.grid[r][c] = str(crop["crop_tile"])
        self.state.tile_interactions[(r, c)] = 0
        self.state.plant_plots[(r, c)] = PlantPlotState(
            crop_id=crop_id,
            planter_id=actor_id,
            planter_faction_id=int(actor.faction_id),
        )
        farming = self._skill_level(actor, "farming")
        preserve_chance = min(0.5, 0.05 * float(farming))
        if self._rng.random() < preserve_chance:
            actor.inventory.append(seed_item)
            events.append(f"plant:seed_preserved:{seed_item}")
        self._gain_skill_xp(actor, "farming", 2, events)
        events.append(f"plant:{crop_id}:{r}:{c}")
        return 0.08 + min(0.08, 0.01 * float(farming)), True

    def _interact_harvest_plant(
        self, actor: AgentState, actor_id: str, events: List[str]
    ) -> Tuple[float, bool]:
        assert self.state is not None
        pos = actor.position
        if self._harvest_exhausted(pos, self.state.grid[pos[0]][pos[1]], events):
            return -0.02, True
        plot = self.state.plant_plots.get(pos)
        crop_id = plot.crop_id if plot is not None else self._crop_from_tile(self.state.grid[pos[0]][pos[1]])
        if not crop_id or crop_id not in PLANT_TYPES:
            return 0.0, False
        crop = PLANT_TYPES[crop_id]
        food_item = str(crop["food_item"])
        seed_item = str(crop["seed_item"])
        food_min, food_max = tuple(crop["food_qty"])
        seed_min, seed_max = tuple(crop["seed_qty"])
        farming = self._skill_level(actor, "farming")
        food_qty = self._rng.randint(int(food_min), int(food_max)) + (farming // 4)
        seed_qty = self._rng.randint(int(seed_min), int(seed_max)) + (farming // 5)
        food_added = self._add_item_or_drop(actor, food_item, max(1, int(food_qty)), events)
        seed_added = self._add_item_or_drop(actor, seed_item, max(1, int(seed_qty)), events)
        self.state.grid[pos[0]][pos[1]] = (
            self.mapgen_cfg.floor_fallback_id
            if self.mapgen_cfg.floor_fallback_id in self.tiles
            else "floor"
        )
        self._record_harvest(pos)
        self.state.plant_plots.pop(pos, None)
        self.state.tile_interactions.pop(pos, None)
        same_faction_bonus = 0.0
        if (
            plot is not None
            and int(actor.faction_id) >= 0
            and int(plot.planter_faction_id) >= 0
            and int(actor.faction_id) == int(plot.planter_faction_id)
        ):
            same_faction_bonus = 0.12
            events.append(
                f"harvest:faction_bonus:{crop_id}:planter={plot.planter_id}:faction={plot.planter_faction_id}"
            )
        self._gain_skill_xp(actor, "farming", max(1, 2 + int(food_qty) + int(seed_qty)), events)
        events.append(
            f"harvest:{crop_id}:food={food_item}:{food_qty}:{food_added}:seed={seed_item}:{seed_qty}:{seed_added}"
        )
        base = 0.08 + 0.02 * float(max(1, int(food_qty)))
        return base + same_faction_bonus, True

    def _first_plantable_crop(self, inventory: List[str]) -> str:
        counts = self._count_base_items(inventory)
        for crop_id, row in sorted(PLANT_TYPES.items()):
            seed_item = str(row.get("seed_item", "")).strip()
            if seed_item and int(counts.get(seed_item, 0)) > 0:
                return crop_id
        return ""

    def _crop_from_tile(self, tile_id: str) -> str:
        for crop_id, row in PLANT_TYPES.items():
            if str(row.get("crop_tile", "")).strip() == str(tile_id):
                return crop_id
        return ""

    def _can_carry_item(self, agent: AgentState, item_id: str, qty: int = 1) -> bool:
        base_id = self._item_base_id(item_id)
        extra = float(self.item_weight.get(base_id, 1.0)) * float(max(1, int(qty)))
        cap = self._carry_capacity(agent)
        return (self._carried_weight(agent) + extra) <= (cap * 1.35)

    def _add_item_or_drop(
        self, agent: AgentState, item_id: str, qty: int, events: List[str]
    ) -> int:
        assert self.state is not None
        added = 0
        for _ in range(max(0, int(qty))):
            if self._can_carry_item(agent, item_id, 1):
                agent.inventory.append(item_id)
                added += 1
            else:
                pos = agent.position
                self.state.ground_items.setdefault(pos, []).append(item_id)
                events.append(f"drop_overweight:{item_id}")
        return added

    def _interact_resource_node(
        self, actor: AgentState, actor_id: str, events: List[str]
    ) -> Tuple[float, bool]:
        assert self.state is not None
        node = self.state.resource_nodes.get(actor.position)
        if node is None or int(node.remaining) <= 0:
            return 0.0, False
        if self._harvest_exhausted(actor.position, f"node:{node.node_id}", events):
            return -0.02, True
        skill_level = self._skill_level(actor, node.skill)
        base = self._rng.randint(1, 2)
        tool_bonus = self._max_equipped_tool_bonus(actor, skill=node.skill)
        bonus = int(round(self._agent_enchant_bonus(actor, "gather_yield_plus")))
        qty = min(
            int(node.remaining),
            base + (skill_level // 3) + (tool_bonus // 2) + max(0, bonus),
        )
        if qty <= 0:
            return -0.01, True
        added = self._add_item_or_drop(actor, node.drop_item, qty, events)
        self._record_harvest(actor.position)
        node.remaining = max(0, int(node.remaining) - qty)
        if int(node.remaining) <= 0:
            self.state.resource_nodes.pop(actor.position, None)
            events.append(f"resource_depleted:{node.node_id}")
        self._gain_skill_xp(actor, node.skill, max(1, qty * 2), events)
        events.append(f"gather:{node.node_id}:{node.drop_item}:{qty}:{added}")
        reward = 0.06 * float(max(1, qty))
        if added <= 0:
            reward -= 0.02
            events.append("carry_blocked")
        return reward, True

    def _interact_station(
        self, actor: AgentState, actor_id: str, events: List[str]
    ) -> Tuple[float, bool]:
        assert self.state is not None
        station = self.state.stations.get(actor.position)
        if station is None:
            return 0.0, False
        recipe = self._best_station_recipe(actor, station)
        if recipe is None:
            events.append(f"station_idle:{station.station_id}")
            return -0.01, True
        self._consume_recipe_inputs(actor, recipe)
        if recipe.build_tile_id:
            built = self._build_from_recipe(actor, recipe, events)
            if not built:
                # Refund inputs if no valid build location.
                for item_id, qty in recipe.inputs.items():
                    self._add_item_or_drop(actor, item_id, qty, events)
                events.append("build_fail:no_space")
                return -0.02, True
        else:
            for item_id, qty in recipe.outputs.items():
                bonus = max(0, int(station.quality_tier))
                added_qty = self._add_item_or_drop(actor, item_id, int(qty) + bonus, events)
                if added_qty > 0 and int(self.base_durability_by_item.get(item_id, 0)) > 0:
                    crafting_level = self._skill_level(actor, "crafting")
                    created = 0
                    for idx in range(len(actor.inventory) - 1, -1, -1):
                        if created >= added_qty:
                            break
                        if self._item_base_id(actor.inventory[idx]) != item_id:
                            continue
                        self._ensure_durable_item_instance(
                            actor.inventory, idx, crafting_skill=crafting_level
                        )
                        created += 1
        craft_skill = recipe.skill or "crafting"
        self._gain_skill_xp(actor, craft_skill, max(1, 2 + recipe.min_skill), events)
        required_tool = str(recipe.required_tool_category).strip().lower()
        if required_tool:
            used_tool = self._equipped_tool_for_category(actor, required_tool)
            if used_tool is not None:
                self._consume_item_durability(
                    actor,
                    used_tool,
                    TOOL_DURABILITY_USE_BY_CATEGORY.get(required_tool, 1),
                    events,
                )
        speed = max(0.1, float(station.speed_multiplier) * float(recipe.speed_multiplier))
        events.append(f"craft:{recipe.recipe_id}:station={station.station_id}:speed={speed:.2f}")
        return 0.14 + (0.02 * float(station.quality_tier)), True

    def _consume_recipe_inputs(self, agent: AgentState, recipe: RecipeDef) -> None:
        for item_id, qty in recipe.inputs.items():
            remaining = int(qty)
            for idx in range(len(agent.inventory) - 1, -1, -1):
                if remaining <= 0:
                    break
                if self._item_base_id(agent.inventory[idx]) == item_id:
                    agent.inventory.pop(idx)
                    remaining -= 1

    def _best_station_recipe(
        self, agent: AgentState, station: StationState
    ) -> RecipeDef | None:
        allowed = set(station.unlock_recipes)
        candidates: List[RecipeDef] = []
        for rid in self._recipe_ids:
            recipe = self.recipes[rid]
            if recipe.station and recipe.station != station.station_id:
                continue
            if allowed and rid not in allowed:
                continue
            if self._skill_level(agent, recipe.skill or "crafting") < recipe.min_skill:
                continue
            if not self._has_required_recipe_tool(agent, recipe):
                continue
            if not self._has_recipe_inputs(agent, recipe):
                continue
            candidates.append(recipe)
        if not candidates:
            return None
        candidates.sort(key=lambda r: (r.min_skill, r.recipe_id), reverse=True)
        return candidates[0]

    def _has_recipe_inputs(self, agent: AgentState, recipe: RecipeDef) -> bool:
        counts = self._count_base_items(agent.inventory)
        for item_id, qty in recipe.inputs.items():
            if int(counts.get(item_id, 0)) < int(qty):
                return False
        return True

    def _build_from_recipe(
        self, actor: AgentState, recipe: RecipeDef, events: List[str]
    ) -> bool:
        assert self.state is not None
        if recipe.build_tile_id not in self.tiles:
            return False
        ar, ac = actor.position
        for nr, nc in ((ar - 1, ac), (ar + 1, ac), (ar, ac - 1), (ar, ac + 1)):
            if nr <= 0 or nc <= 0:
                continue
            if nr >= (len(self.state.grid) - 1) or nc >= (len(self.state.grid[0]) - 1):
                continue
            occupied = any(a.alive and a.position == (nr, nc) for a in self.state.agents.values())
            occupied = occupied or any(
                m.alive and m.position == (nr, nc) for m in self.state.monsters.values()
            )
            occupied = occupied or ((nr, nc) in self.state.chests)
            occupied = occupied or ((nr, nc) in self.state.stations)
            occupied = occupied or ((nr, nc) in self.state.resource_nodes)
            if occupied:
                continue
            self.state.grid[nr][nc] = recipe.build_tile_id
            if recipe.build_tile_id == "campfire":
                self.state.tile_interactions[(nr, nc)] = max(
                    1, int(self.state.tile_interactions.get((nr, nc), 0)) + FIRE_FUEL_PER_STICK
                )
            events.append(f"build:{recipe.build_tile_id}:{nr}:{nc}")
            return True
        return False

    def _item_base_id(self, item_id: str) -> str:
        assert self.state is not None
        meta = self.state.item_metadata.get(str(item_id), {})
        base = str(meta.get("base_id", "")).strip()
        return base or str(item_id)

    def _item_current_durability(self, item_id: str) -> int:
        assert self.state is not None
        row = self.state.item_metadata.get(str(item_id), {})
        if "durability" in row:
            return int(row.get("durability", 0))
        base = self._item_base_id(item_id)
        return int(self.base_durability_by_item.get(base, 0))

    def _ensure_durable_item_instance(
        self, bag: List[str], index: int, *, crafting_skill: int = 0
    ) -> str:
        assert self.state is not None
        item_id = str(bag[index])
        base = self._item_base_id(item_id)
        max_durability = int(self.base_durability_by_item.get(base, 0))
        if max_durability <= 0:
            return item_id
        if item_id in self.state.item_metadata and "durability" in self.state.item_metadata[item_id]:
            return item_id
        bonus = max(0, int(crafting_skill) // 2)
        token = self._make_item_instance(
            base_id=base,
            kind="durable_item",
            metadata={
                "durability": max_durability + bonus,
                "max_durability": max_durability + bonus,
                "crafted_by_skill": int(crafting_skill),
            },
        )
        bag[index] = token
        return token

    def _consume_item_durability(
        self, agent: AgentState, equipped_item: str, amount: int, events: List[str]
    ) -> None:
        assert self.state is not None
        base = self._item_base_id(equipped_item)
        max_durability = int(self.base_durability_by_item.get(base, 0))
        if max_durability <= 0:
            return
        for idx in range(len(agent.equipped) - 1, -1, -1):
            if agent.equipped[idx] != equipped_item:
                continue
            token = self._ensure_durable_item_instance(agent.equipped, idx)
            row = self.state.item_metadata.setdefault(token, {"base_id": base, "kind": "durable_item"})
            row["durability"] = max(0, int(row.get("durability", max_durability)) - max(1, int(amount)))
            self.state.item_metadata[token] = row
            events.append(f"durability_use:{base}:{int(row['durability'])}")
            if int(row["durability"]) <= 0:
                broken = agent.equipped.pop(idx)
                for slot_name, slot_item in list(agent.armor_slots.items()):
                    if slot_item == broken:
                        agent.armor_slots[slot_name] = None
                self.state.item_metadata.pop(broken, None)
                events.append(f"item_broken:{base}")
            return

    def _max_equipped_tool_bonus(
        self,
        agent: AgentState,
        *,
        skill: str = "",
        category: str = "",
    ) -> int:
        best = 0
        skill = str(skill).strip().lower()
        category = str(category).strip().lower()
        for item in agent.equipped:
            base = self._item_base_id(item)
            tool_category = str(self.tool_category_by_item.get(base, "")).strip().lower()
            if category and tool_category != category:
                continue
            tool_skill = str(self.tool_skill_by_item.get(base, "")).strip().lower()
            if skill and tool_skill != skill:
                continue
            best = max(best, int(self.tool_skill_bonus_by_item.get(base, 0)))
        return max(0, int(best))

    def _has_required_recipe_tool(self, agent: AgentState, recipe: RecipeDef) -> bool:
        required = str(recipe.required_tool_category).strip().lower()
        if not required:
            return True
        for item in agent.equipped:
            base = self._item_base_id(item)
            if str(self.tool_category_by_item.get(base, "")).strip().lower() == required:
                return True
        return False

    def _equipped_tool_for_category(self, agent: AgentState, category: str) -> str | None:
        required = str(category).strip().lower()
        if not required:
            return None
        for item in reversed(agent.equipped):
            base = self._item_base_id(item)
            if str(self.tool_category_by_item.get(base, "")).strip().lower() == required:
                return item
        return None

    def _harvest_exhausted(self, pos: Tuple[int, int], tile_id: str, events: List[str]) -> bool:
        assert self.state is not None
        used = int(self.state.tile_harvest_counts.get(pos, 0))
        if used >= HARVESTS_PER_TILE_LIMIT:
            events.append(f"harvest_tile_exhausted:{tile_id}:{pos[0]}:{pos[1]}")
            return True
        return False

    def _record_harvest(self, pos: Tuple[int, int], amount: int = 1) -> None:
        assert self.state is not None
        self.state.tile_harvest_counts[pos] = int(self.state.tile_harvest_counts.get(pos, 0)) + max(1, int(amount))

    def _pop_first_base_item(self, bag: List[str], base_id: str) -> str | None:
        for idx, item in enumerate(bag):
            if self._item_base_id(item) == base_id:
                return bag.pop(idx)
        return None

    def _count_base_items(self, bag: List[str]) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for item in bag:
            base = self._item_base_id(item)
            out[base] = out.get(base, 0) + 1
        return out

    def _make_item_instance(
        self,
        *,
        base_id: str,
        kind: str,
        metadata: Dict[str, object],
    ) -> str:
        assert self.state is not None
        uid = int(self.state.item_metadata.get("__next_uid__", {}).get("value", 1))
        self.state.item_metadata["__next_uid__"] = {"value": uid + 1}
        token = f"{kind}@{uid}"
        row = dict(metadata)
        row["base_id"] = str(base_id)
        row["kind"] = str(kind)
        self.state.item_metadata[token] = row
        return token

    def _is_skill_book(self, item_id: str) -> bool:
        assert self.state is not None
        return str(self.state.item_metadata.get(str(item_id), {}).get("kind", "")) == "skill_book"

    def _item_enchant_rows(self, item_id: str) -> List[Dict[str, object]]:
        assert self.state is not None
        row = self.state.item_metadata.get(str(item_id), {})
        raw = row.get("enchantments", [])
        if not isinstance(raw, list):
            return []
        return [dict(x) for x in raw if isinstance(x, dict)]

    def _enchant_bonus_for_item(self, item_id: str, bonus_key: str) -> float:
        total = 0.0
        for ench in self._item_enchant_rows(item_id):
            eid = str(ench.get("id", "")).strip()
            edef = self.enchant_defs.get(eid)
            if edef is None:
                continue
            for eff in edef.effects:
                if str(eff.get("type", "")) == bonus_key:
                    total += float(eff.get("amount", 0.0))
        return total

    def _agent_enchant_bonus(self, agent: AgentState, bonus_key: str) -> float:
        total = 0.0
        for item in agent.equipped:
            total += self._enchant_bonus_for_item(item, bonus_key)
        return total

    def _use_item_or_spell(self, actor: AgentState, aid: str, events: List[str]) -> float:
        assert self.state is not None
        medic_level = self._skill_level(actor, "medic")
        rewards = {aid: 0.0}
        info = {aid: {"events": events}}
        bandage_item = self._pop_first_base_item(actor.inventory, "bandage")
        if bandage_item is not None and actor.hp < actor.max_hp:
            heal = 3 + (medic_level // 2) + max(0, (actor.intellect - 5) // 4)
            actor.hp = min(actor.max_hp, actor.hp + heal)
            events.append("use:bandage")
            self._clear_status(aid, "bleed", info=info, rewards=rewards)
            self._gain_skill_xp(actor, "medic", 2, events)
            return 0.12 + 0.01 * medic_level
        if bandage_item is not None:
            actor.inventory.insert(0, bandage_item)

        heal_item = self._pop_first_base_item(actor.inventory, "healing_potion")
        if heal_item is not None and actor.hp < actor.max_hp:
            heal = 5 + medic_level + max(0, (actor.intellect - 5) // 3)
            actor.hp = min(actor.max_hp, actor.hp + heal)
            events.append("use:healing_potion")
            self._gain_skill_xp(actor, "medic", 3, events)
            return 0.2 + 0.015 * medic_level
        if heal_item is not None:
            actor.inventory.insert(0, heal_item)

        if self._pop_first_base_item(actor.inventory, "antidote") is not None:
            self._clear_status(aid, "poison", info=info, rewards=rewards)
            events.append("use:antidote")
            self._gain_skill_xp(actor, "medic", 2, events)
            return 0.08
        if self._pop_first_base_item(actor.inventory, "cleanse_potion") is not None:
            for sid in ("poison", "bleed", "confused", "paralyzed"):
                self._clear_status(aid, sid, info=info, rewards=rewards)
            events.append("use:cleanse_potion")
            self._gain_skill_xp(actor, "alchemy", 2, events)
            return 0.1
        if self._pop_first_base_item(actor.inventory, "regen_potion") is not None:
            self._apply_status(
                target_aid=aid,
                status_id="regen",
                source_aid=aid,
                info=info,
                rewards=rewards,
            )
            events.append("use:regen_potion")
            self._gain_skill_xp(actor, "alchemy", 2, events)
            return 0.08
        if self._pop_first_base_item(actor.inventory, "resistance_tonic") is not None:
            self._apply_status(
                target_aid=aid,
                status_id="resistance",
                source_aid=aid,
                info=info,
                rewards=rewards,
            )
            events.append("use:resistance_tonic")
            self._gain_skill_xp(actor, "alchemy", 2, events)
            return 0.08
        if (
            self._pop_first_base_item(actor.inventory, "blank_book") is not None
            and self._pop_first_base_item(actor.inventory, "ink_vial") is not None
        ):
            skill = self._best_teachable_skill(actor)
            book_cap = max(1, self._skill_level(actor, skill) - 1)
            token = self._make_item_instance(
                base_id="skill_book",
                kind="skill_book",
                metadata={
                    "skill_name": skill,
                    "max_teachable_level": book_cap,
                    "uses": 3,
                    "author_id": aid,
                    "author_level": self._skill_level(actor, skill),
                },
            )
            actor.inventory.append(token)
            events.append(f"write_book:{skill}:cap={book_cap}")
            self._gain_skill_xp(actor, "crafting", 3, events)
            return 0.1
        if self._pop_first_base_item(actor.inventory, "enchant_rune") is not None:
            if self._apply_enchant_to_equipped(actor, aid, events):
                self._gain_skill_xp(actor, "crafting", 2, events)
                return 0.11
            actor.inventory.append("enchant_rune")
        spell_reward, casted = self._cast_best_spell(actor, aid, info=info, rewards=rewards)
        if casted:
            return spell_reward
        return -0.01

    def _best_teachable_skill(self, agent: AgentState) -> str:
        candidates = [
            "mining",
            "woodcutting",
            "crafting",
            "smithing",
            "alchemy",
            "farming",
            "medic",
            "athletics",
        ]
        best = "crafting"
        best_level = -1
        for skill in candidates:
            lvl = self._skill_level(agent, skill)
            if lvl > best_level:
                best = skill
                best_level = lvl
        return best

    def _apply_enchant_to_equipped(self, actor: AgentState, aid: str, events: List[str]) -> bool:
        assert self.state is not None
        if not actor.equipped:
            events.append("enchant_fail:no_equipped")
            return False
        item = actor.equipped[-1]
        base = self._item_base_id(item)
        choice = "status_resist"
        if base in self.weapon_damage_type:
            choice = "damage_plus"
        elif base in self.armor_slot_by_item:
            choice = "defense_plus"
        elif str(self.tool_category_by_item.get(base, "")).strip() in {"axe", "pickaxe", "shovel", "handaxe"}:
            choice = "gather_yield_plus"
        meta = self.state.item_metadata.setdefault(str(item), {"base_id": base})
        ench = list(meta.get("enchantments", []))
        limits = self.enchant_defs.get(choice)
        max_stacks = int(limits.max_stacks) if limits is not None else 1
        current = sum(1 for row in ench if str(row.get("id", "")) == choice)
        if current >= max_stacks:
            events.append(f"enchant_fail:max_stacks:{choice}")
            return False
        ench.append({"id": choice, "by": aid})
        meta["enchantments"] = ench
        meta["kind"] = str(meta.get("kind", "enchanted_item"))
        meta["base_id"] = base
        self.state.item_metadata[str(item)] = meta
        events.append(f"enchant:{choice}:{base}:by={aid}")
        return True

    def _cast_best_spell(
        self,
        actor: AgentState,
        aid: str,
        info: Dict[str, Dict[str, object]],
        rewards: Dict[str, float],
    ) -> Tuple[float, bool]:
        assert self.state is not None
        for spell_id in actor.known_spells:
            spell = self.spell_defs.get(spell_id)
            if spell is None:
                continue
            if int(actor.mana) < int(spell.mana_cost):
                continue
            if int(actor.spell_cooldowns.get(spell_id, 0)) > 0:
                continue
            if not self._has_required_reagents(actor.inventory, spell):
                continue
            target_id = self._choose_spell_target(aid, spell)
            if target_id is None:
                continue
            self._consume_required_reagents(actor.inventory, spell)
            actor.mana = max(0, int(actor.mana) - int(spell.mana_cost))
            if spell.cooldown > 0:
                actor.spell_cooldowns[spell_id] = int(spell.cooldown)
            source = actor
            target = self.state.agents[target_id]
            delta = self._apply_effects(
                effects=spell.effects,
                source_agent=source,
                target_agent=target,
                events=info[aid]["events"],
                rewards=rewards,
                target_aid=target_id,
                context=f"spell:{spell_id}",
            )
            info[aid]["events"].append(f"cast:{spell_id}:{target_id}")
            self._gain_skill_xp(actor, "alchemy", 1, info[aid]["events"])
            return 0.05 + delta, True
        return 0.0, False

    def _has_required_reagents(self, inventory: List[str], spell: SpellDef) -> bool:
        counts = self._count_base_items(inventory)
        for item_id, qty in spell.required_reagents.items():
            if int(counts.get(item_id, 0)) < int(qty):
                return False
        return True

    def _consume_required_reagents(self, inventory: List[str], spell: SpellDef) -> None:
        for item_id, qty in spell.required_reagents.items():
            for _ in range(int(qty)):
                taken = self._pop_first_base_item(inventory, item_id)
                if taken is None:
                    break

    def _choose_spell_target(self, aid: str, spell: SpellDef) -> str | None:
        assert self.state is not None
        if spell.target == "self":
            return aid
        actor = self.state.agents[aid]
        if spell.target == "ally":
            allies = [
                other_id
                for other_id, other in self.state.agents.items()
                if other_id != aid
                and other.alive
                and self._is_allied(actor, other)
                and self._manhattan(actor.position, other.position) <= max(1, int(spell.range))
            ]
            return sorted(allies)[0] if allies else None
        enemies = [
            other_id
            for other_id, other in self.state.agents.items()
            if other_id != aid
            and other.alive
            and not self._is_allied(actor, other)
            and self._manhattan(actor.position, other.position) <= max(1, int(spell.range))
        ]
        if enemies:
            enemies.sort(
                key=lambda x: self._manhattan(actor.position, self.state.agents[x].position)
            )
            return enemies[0]
        return None

    def _status_has_tag(self, aid: str, tag: str) -> bool:
        assert self.state is not None
        for active in self.state.agent_statuses.get(aid, []):
            sdef = self.status_defs.get(active.status_id)
            if sdef is None:
                continue
            if tag in sdef.tags:
                return True
        return False

    def _status_adjust_action(
        self, aid: str, action: int, info: Dict[str, Dict[str, object]]
    ) -> int:
        if self.state is None:
            return action
        if self._status_has_tag(aid, "paralyze"):
            info[aid]["events"].append("status_block:paralyzed")
            return ACTION_WAIT
        if self._status_has_tag(aid, "confuse"):
            roll = self._rng.random()
            if roll < 0.4:
                choices = [ACTION_MOVE_NORTH, ACTION_MOVE_SOUTH, ACTION_MOVE_WEST, ACTION_MOVE_EAST, ACTION_WAIT]
                scrambled = int(self._rng.choice(choices))
                info[aid]["events"].append(f"status_confused_action:{action}->{scrambled}")
                return scrambled
        return action

    def _tick_agent_statuses(
        self,
        aid: str,
        rewards: Dict[str, float],
        info: Dict[str, Dict[str, object]],
    ) -> float:
        assert self.state is not None
        agent = self.state.agents[aid]
        bag = list(self.state.agent_statuses.get(aid, []))
        keep: List[ActiveStatus] = []
        delta = 0.0
        for active in bag:
            sdef = self.status_defs.get(active.status_id)
            if sdef is None:
                continue
            active.remaining = int(active.remaining) - 1
            active.tick_counter = int(active.tick_counter) + 1
            if active.tick_counter >= max(1, int(active.tick_interval)):
                active.tick_counter = 0
                delta += self._apply_effects(
                    effects=sdef.tick_effects,
                    source_agent=self.state.agents.get(active.source_id),
                    target_agent=agent,
                    events=info[aid]["events"],
                    rewards=rewards,
                    target_aid=aid,
                    context="status_tick",
                )
            if active.remaining <= 0:
                delta += self._apply_effects(
                    effects=sdef.expire_effects,
                    source_agent=self.state.agents.get(active.source_id),
                    target_agent=agent,
                    events=info[aid]["events"],
                    rewards=rewards,
                    target_aid=aid,
                    context="status_expire",
                )
                info[aid]["events"].append(f"status_expire:{active.status_id}")
                continue
            keep.append(active)
        self.state.agent_statuses[aid] = keep
        return delta

    def _apply_status(
        self,
        target_aid: str,
        status_id: str,
        source_aid: str,
        info: Dict[str, Dict[str, object]],
        rewards: Dict[str, float],
    ) -> None:
        assert self.state is not None
        sdef = self.status_defs.get(status_id)
        if sdef is None:
            return
        agent = self.state.agents[target_aid]
        resist = 0.0
        if self._status_has_tag(target_aid, "status_resist"):
            resist += 0.25
        resist += self._agent_enchant_bonus(agent, "status_resist")
        if resist > 0 and self._rng.random() < min(0.85, resist):
            info[target_aid]["events"].append(f"status_resisted:{status_id}")
            return
        bag = self.state.agent_statuses.setdefault(target_aid, [])
        bag = [x for x in bag if x.status_id != status_id]
        bag.append(
            ActiveStatus(
                status_id=status_id,
                remaining=max(1, int(sdef.duration)),
                tick_interval=max(1, int(sdef.tick_interval)),
                tick_counter=0,
                source_id=str(source_aid),
            )
        )
        self.state.agent_statuses[target_aid] = bag
        info[target_aid]["events"].append(f"status_apply:{status_id}:{source_aid}")
        self._apply_effects(
            effects=sdef.apply_effects,
            source_agent=self.state.agents.get(source_aid),
            target_agent=agent,
            events=info[target_aid]["events"],
            rewards=rewards,
            target_aid=target_aid,
            context="status_apply",
        )

    def _clear_status(
        self,
        target_aid: str,
        status_id: str,
        info: Dict[str, Dict[str, object]],
        rewards: Dict[str, float],
    ) -> bool:
        assert self.state is not None
        sdef = self.status_defs.get(status_id)
        if sdef is None:
            return False
        bag = self.state.agent_statuses.setdefault(target_aid, [])
        kept = [x for x in bag if x.status_id != status_id]
        changed = len(kept) != len(bag)
        self.state.agent_statuses[target_aid] = kept
        if changed:
            agent = self.state.agents[target_aid]
            self._apply_effects(
                effects=sdef.expire_effects,
                source_agent=agent,
                target_agent=agent,
                events=info[target_aid]["events"],
                rewards=rewards,
                target_aid=target_aid,
                context="status_cleanse",
            )
            info[target_aid]["events"].append(f"status_cleared:{status_id}")
        return changed

    def _apply_effects(
        self,
        effects: List[Dict[str, object]],
        source_agent: AgentState | None,
        target_agent: AgentState,
        events: List[str],
        rewards: Dict[str, float],
        target_aid: str,
        context: str,
    ) -> float:
        delta = 0.0
        for eff in effects:
            et = str(eff.get("type", "")).strip()
            amount = float(eff.get("amount", 0.0))
            if et == "damage":
                dmg = max(0, int(round(amount)))
                if dmg > 0:
                    target_agent.hp = max(0, int(target_agent.hp) - dmg)
                    rewards[target_aid] = float(rewards.get(target_aid, 0.0)) - (0.02 * float(dmg))
                    delta -= 0.02 * float(dmg)
                    events.append(f"{context}:damage:{dmg}")
            elif et == "heal":
                heal = max(0, int(round(amount)))
                if heal > 0:
                    target_agent.hp = min(int(target_agent.max_hp), int(target_agent.hp) + heal)
                    rewards[target_aid] = float(rewards.get(target_aid, 0.0)) + (0.01 * float(heal))
                    delta += 0.01 * float(heal)
                    events.append(f"{context}:heal:{heal}")
            elif et == "hunger":
                gain = int(round(amount))
                target_agent.hunger = max(0, min(int(target_agent.max_hunger), int(target_agent.hunger) + gain))
                events.append(f"{context}:hunger:{gain}")
            elif et == "mana":
                gain = int(round(amount))
                target_agent.mana = max(0, min(int(target_agent.max_mana), int(target_agent.mana) + gain))
                events.append(f"{context}:mana:{gain}")
            elif et == "apply_status":
                status_id = str(eff.get("status", "")).strip()
                if status_id and self.state is not None:
                    source_id = source_agent.agent_id if source_agent is not None else target_aid
                    tmp_info = {target_aid: {"events": events}}
                    self._apply_status(
                        target_aid=target_aid,
                        status_id=status_id,
                        source_aid=source_id,
                        info=tmp_info,
                        rewards=rewards,
                    )
            elif et == "cure_status":
                status_id = str(eff.get("status", "")).strip()
                if status_id:
                    tmp_info = {target_aid: {"events": events}}
                    self._clear_status(
                        target_aid=target_aid,
                        status_id=status_id,
                        info=tmp_info,
                        rewards=rewards,
                    )
            elif et == "cleanse":
                tmp_info = {target_aid: {"events": events}}
                for sid in ("poison", "bleed", "confused", "paralyzed"):
                    self._clear_status(
                        target_aid=target_aid,
                        status_id=sid,
                        info=tmp_info,
                        rewards=rewards,
                    )
            elif et == "reveal":
                events.append(f"{context}:reveal")
            elif et == "teleport_blink":
                if self.state is not None:
                    tr, tc = target_agent.position
                    options = [
                        (tr - 1, tc),
                        (tr + 1, tc),
                        (tr, tc - 1),
                        (tr, tc + 1),
                    ]
                    walkable = [(r, c) for (r, c) in options if self._walkable(r, c)]
                    if walkable:
                        old = target_agent.position
                        target_agent.position = self._rng.choice(walkable)
                        events.append(f"{context}:blink:{old}->{target_agent.position}")
            elif et == "knockback":
                if self.state is not None and source_agent is not None:
                    sr, sc = source_agent.position
                    tr, tc = target_agent.position
                    dr = 1 if tr > sr else (-1 if tr < sr else 0)
                    dc = 1 if tc > sc else (-1 if tc < sc else 0)
                    nr, nc = tr + dr, tc + dc
                    if self._walkable(nr, nc):
                        old = target_agent.position
                        target_agent.position = (nr, nc)
                        events.append(f"{context}:knockback:{old}->{target_agent.position}")
        return delta

    def _handle_faction_interact(
        self, actor: AgentState, actor_id: str, events: List[str]
    ) -> Tuple[float, bool]:
        assert self.state is not None
        reward = 0.0
        pending = self.state.pending_faction_invites.get(actor_id)
        if pending and actor.faction_id < 0:
            if self._invite_is_expired(pending):
                self.state.pending_faction_invites.pop(actor_id, None)
                events.append("faction_invite_expired")
                return (
                    -float(self.config.invalid_faction_action_penalty),
                    True,
                )
            if self._action_cooldown_blocked(actor_id, "join_faction", events):
                return -float(self.config.invalid_faction_action_penalty), True
            inviter_id = str(pending.get("inviter_id", ""))
            faction_id = int(pending.get("faction_id", -1))
            inviter = self.state.agents.get(inviter_id)
            if inviter is None or not inviter.alive or inviter.faction_id != faction_id:
                self.state.pending_faction_invites.pop(actor_id, None)
                events.append("faction_invite_expired")
                return -float(self.config.invalid_faction_action_penalty), True
            if self._manhattan(actor.position, inviter.position) == 1:
                actor.faction_id = faction_id
                self.state.pending_faction_invites.pop(actor_id, None)
                events.append(f"faction_join:{faction_id}:{inviter_id}")
                reward += float(self.config.team_join_reward)
                reward += self._award_join_side_bonuses(
                    inviter_id=inviter_id, invitee_id=actor_id, faction_id=faction_id
                )
                self._set_action_cooldown(actor_id, "join_faction")
                return reward, True

        if actor.faction_id < 0:
            if self._action_cooldown_blocked(actor_id, "create_faction", events):
                return -float(self.config.invalid_faction_action_penalty), True
            new_faction = int(self._next_faction_id)
            self._next_faction_id += 1
            actor.faction_id = new_faction
            self.state.faction_leaders[new_faction] = actor_id
            events.append(f"faction_create:{new_faction}")
            reward += float(self.config.team_create_reward)
            self._set_action_cooldown(actor_id, "create_faction")
            return reward, True

        if not self._is_faction_leader(actor_id):
            return 0.0, False
        if self._action_cooldown_blocked(actor_id, "invite_faction", events):
            return -float(self.config.invalid_faction_action_penalty), True
        target_id = self._adjacent_invite_candidate(actor_id)
        if target_id is None:
            return 0.0, False
        target = self.state.agents[target_id]
        if target.faction_id == actor.faction_id:
            return 0.0, False

        self.state.pending_faction_invites[target_id] = {
            "faction_id": int(actor.faction_id),
            "inviter_id": actor_id,
            "created_step": int(self.state.step_count),
        }
        events.append(f"faction_invite_sent:{target_id}:{actor.faction_id}")
        reward += float(self.config.team_invite_reward)
        self._set_action_cooldown(actor_id, "invite_faction")
        return reward, True

    def _adjacent_ally_ids(self, actor_id: str) -> List[str]:
        assert self.state is not None
        actor = self.state.agents[actor_id]
        out: List[str] = []
        for other_id in self.possible_agents:
            if other_id == actor_id:
                continue
            other = self.state.agents[other_id]
            if not other.alive:
                continue
            if not self._is_allied(actor, other):
                continue
            if self._manhattan(actor.position, other.position) != 1:
                continue
            out.append(other_id)
        return out

    def _give_to_ally(self, actor: AgentState, actor_id: str, events: List[str]) -> float:
        assert self.state is not None
        if self._action_cooldown_blocked(actor_id, "team_give", events):
            return -float(self.config.invalid_faction_action_penalty)
        candidates = self._adjacent_ally_ids(actor_id)
        if not candidates:
            events.append("team_give_fail:no_ally")
            return -0.01
        if not actor.inventory:
            events.append("team_give_fail:no_item")
            return -0.01
        target_id = sorted(candidates)[0]
        target = self.state.agents[target_id]
        item = actor.inventory.pop(0)
        if self._is_skill_book(item):
            taught = self._teach_from_book(
                teacher=actor,
                teacher_id=actor_id,
                target=target,
                target_id=target_id,
                book_item=item,
                events=events,
            )
            if not taught:
                target.inventory.append(item)
                events.append(f"team_give:{target_id}:{item}")
        else:
            target.inventory.append(item)
            events.append(f"team_give:{target_id}:{item}")
        self._set_action_cooldown(actor_id, "team_give")
        allowed = self._team_pair_reward_allowed(
            action="team_give",
            a=actor_id,
            b=target_id,
            cap=max(0, int(self.config.team_pair_reward_cap_per_episode)),
            repeat_guard=max(0, int(self.config.team_pair_reward_repeat_guard_steps)),
            events=events,
        )
        if not allowed:
            return 0.0
        return float(self.config.team_give_reward)

    def _teach_from_book(
        self,
        teacher: AgentState,
        teacher_id: str,
        target: AgentState,
        target_id: str,
        book_item: str,
        events: List[str],
    ) -> bool:
        assert self.state is not None
        meta = dict(self.state.item_metadata.get(book_item, {}))
        if not meta:
            return False
        skill = str(meta.get("skill_name", "")).strip()
        if not skill:
            return False
        book_cap = max(0, int(meta.get("max_teachable_level", 0)))
        author_level = max(0, int(meta.get("author_level", self._skill_level(teacher, skill))))
        margin = 1
        cap = min(book_cap, max(0, author_level - margin))
        current = self._skill_level(target, skill)
        if current >= cap:
            events.append(f"teach_fail:cap_reached:{target_id}:{skill}:{cap}")
            target.inventory.append(book_item)
            return False
        target.skills[skill] = current + 1
        uses = max(0, int(meta.get("uses", 1)) - 1)
        meta["uses"] = uses
        self.state.item_metadata[book_item] = meta
        events.append(f"teach:{target_id}:{skill}:{current + 1}:by={teacher_id}")
        if uses > 0:
            target.inventory.append(book_item)
        else:
            events.append(f"book_expired:{skill}")
            self.state.item_metadata.pop(book_item, None)
        return True

    def _trade_with_ally(self, actor: AgentState, actor_id: str, events: List[str]) -> float:
        assert self.state is not None
        if self._action_cooldown_blocked(actor_id, "team_trade", events):
            return -float(self.config.invalid_faction_action_penalty)
        candidates = self._adjacent_ally_ids(actor_id)
        if not candidates:
            events.append("team_trade_fail:no_ally")
            return -0.01
        target_id = sorted(candidates)[0]
        target = self.state.agents[target_id]
        if not actor.inventory or not target.inventory:
            events.append("team_trade_fail:missing_item")
            return -0.01
        actor_item = actor.inventory.pop(0)
        target_item = target.inventory.pop(0)
        actor.inventory.append(target_item)
        target.inventory.append(actor_item)
        events.append(f"team_trade:{target_id}:{actor_item}<->{target_item}")
        self._set_action_cooldown(actor_id, "team_trade")
        allowed = self._team_pair_reward_allowed(
            action="team_trade",
            a=actor_id,
            b=target_id,
            cap=max(0, int(self.config.team_pair_reward_cap_per_episode)),
            repeat_guard=max(0, int(self.config.team_pair_reward_repeat_guard_steps)),
            events=events,
        )
        if not allowed:
            return 0.0
        return float(self.config.team_trade_reward)

    def _adjacent_dead_ally_ids(self, actor_id: str) -> List[str]:
        assert self.state is not None
        actor = self.state.agents[actor_id]
        out: List[str] = []
        for other_id in self.possible_agents:
            if other_id == actor_id:
                continue
            other = self.state.agents[other_id]
            if other.alive:
                continue
            if not self._is_allied(actor, other):
                continue
            if self._manhattan(actor.position, other.position) != 1:
                continue
            out.append(other_id)
        return out

    def _revive_ally(self, actor: AgentState, actor_id: str, events: List[str]) -> float:
        assert self.state is not None
        if self._action_cooldown_blocked(actor_id, "team_revive", events):
            return -float(self.config.invalid_faction_action_penalty)
        candidates = self._adjacent_dead_ally_ids(actor_id)
        if not candidates:
            events.append("team_revive_fail:no_dead_ally")
            return -0.02
        if "bandage" in actor.inventory:
            actor.inventory.remove("bandage")
            revive_item = "bandage"
        elif "healing_potion" in actor.inventory:
            actor.inventory.remove("healing_potion")
            revive_item = "healing_potion"
        else:
            events.append("team_revive_fail:no_supplies")
            return -0.02

        target_id = sorted(candidates)[0]
        target = self.state.agents[target_id]
        target.alive = True
        target.hp = max(1, int(target.max_hp) // 2)
        target.hunger = max(0, min(target.max_hunger, int(target.max_hunger) // 4))
        self._revived_this_step.add(target_id)
        events.append(f"team_revive:{target_id}:{revive_item}")
        self._set_action_cooldown(actor_id, "team_revive")
        allowed = self._team_pair_reward_allowed(
            action="team_revive",
            a=actor_id,
            b=target_id,
            cap=max(0, int(self.config.team_pair_reward_cap_per_episode)),
            repeat_guard=max(0, int(self.config.team_pair_reward_repeat_guard_steps)),
            events=events,
        )
        if not allowed:
            return 0.0
        return float(self.config.team_revive_reward)

    def _guard_ally(self, actor: AgentState, actor_id: str, events: List[str]) -> float:
        assert self.state is not None
        if self._action_cooldown_blocked(actor_id, "team_guard", events):
            return -float(self.config.invalid_faction_action_penalty)
        candidates = self._adjacent_ally_ids(actor_id)
        if not candidates:
            events.append("team_guard_fail:no_ally")
            return -0.01
        target_id = sorted(candidates)[0]
        self._guard_assignments[target_id] = actor_id
        events.append(f"team_guard:{target_id}")
        self._set_action_cooldown(actor_id, "team_guard")
        allowed = self._team_pair_reward_allowed(
            action="team_guard",
            a=actor_id,
            b=target_id,
            cap=max(0, int(self.config.team_pair_reward_cap_per_episode)),
            repeat_guard=max(0, int(self.config.team_pair_reward_repeat_guard_steps)),
            events=events,
        )
        if not allowed:
            return 0.0
        return float(self.config.team_guard_reward)

    def _defend(self, _actor: AgentState, actor_id: str, events: List[str]) -> float:
        self._defending_agents.add(actor_id)
        defend_bonus, weapon_id, shield_bonus = self._defend_dr_bonus_for(actor_id)
        events.append(f"defend:dr_bonus:{defend_bonus}")
        events.append(f"defend:weapon:{weapon_id}")
        if shield_bonus > 0:
            events.append(f"defend:shield_bonus:{shield_bonus}")
        return 0.01

    def _defend_dr_bonus_for(self, agent_id: str) -> Tuple[int, str, int]:
        if self.state is None:
            return max(0, int(self.config.defend_unarmed_dr_bonus)), "unarmed", 0
        agent = self.state.agents.get(agent_id)
        if agent is None:
            return max(0, int(self.config.defend_unarmed_dr_bonus)), "unarmed", 0
        weapon_id, _, _, _ = self._equipped_weapon(agent)
        if weapon_id == "unarmed":
            weapon_bonus = max(0, int(self.config.defend_unarmed_dr_bonus))
        else:
            weapon_bonus = max(0, int(self.weapon_defense_dr_bonus.get(weapon_id, 0)))
        shield_bonus = 0
        for item in agent.equipped:
            base = self._item_base_id(item)
            shield_bonus = max(shield_bonus, int(self.item_defense_dr_bonus.get(base, 0)))
        return weapon_bonus + shield_bonus, weapon_id, shield_bonus

    def _leave_faction(self, actor: AgentState, actor_id: str, events: List[str]) -> float:
        assert self.state is not None
        if self._action_cooldown_blocked(actor_id, "leave_faction", events):
            return -float(self.config.invalid_faction_action_penalty)
        old_faction = int(actor.faction_id)
        if old_faction < 0:
            events.append("faction_leave_fail:solo")
            return -float(self.config.invalid_faction_action_penalty)

        was_leader = self.state.faction_leaders.get(old_faction) == actor_id
        actor.faction_id = -1
        self.state.pending_faction_invites.pop(actor_id, None)
        for aid, invite in list(self.state.pending_faction_invites.items()):
            if int(invite.get("faction_id", -1)) == old_faction:
                self.state.pending_faction_invites.pop(aid, None)
        events.append(f"faction_leave:{old_faction}")

        if was_leader:
            candidates = sorted(
                aid
                for aid, a in self.state.agents.items()
                if aid != actor_id and a.alive and int(a.faction_id) == old_faction
            )
            if candidates:
                self.state.faction_leaders[old_faction] = candidates[0]
                events.append(f"faction_new_leader:{old_faction}:{candidates[0]}")
            else:
                self.state.faction_leaders.pop(old_faction, None)
                events.append(f"faction_disband:{old_faction}")
        self._set_action_cooldown(actor_id, "leave_faction")
        return float(self.config.team_leave_reward)

    def _accept_faction_invite(
        self, actor: AgentState, actor_id: str, events: List[str]
    ) -> float:
        assert self.state is not None
        if self._action_cooldown_blocked(actor_id, "join_faction", events):
            return -float(self.config.invalid_faction_action_penalty)
        if int(actor.faction_id) >= 0:
            events.append("faction_accept_fail:already_in_faction")
            return -float(self.config.invalid_faction_action_penalty)

        pending = self.state.pending_faction_invites.get(actor_id)
        if not pending:
            events.append("faction_accept_fail:no_invite")
            return -float(self.config.invalid_faction_action_penalty)
        if self._invite_is_expired(pending):
            self.state.pending_faction_invites.pop(actor_id, None)
            events.append("faction_accept_fail:invite_expired")
            return -float(self.config.invalid_faction_action_penalty)

        inviter_id = str(pending.get("inviter_id", ""))
        faction_id = int(pending.get("faction_id", -1))
        inviter = self.state.agents.get(inviter_id)
        if inviter is None or not inviter.alive or inviter.faction_id != faction_id:
            self.state.pending_faction_invites.pop(actor_id, None)
            events.append("faction_accept_fail:invite_expired")
            return -float(self.config.invalid_faction_action_penalty)
        if self._manhattan(actor.position, inviter.position) != 1:
            events.append("faction_accept_fail:not_adjacent")
            return -float(self.config.invalid_faction_action_penalty)

        actor.faction_id = faction_id
        self.state.pending_faction_invites.pop(actor_id, None)
        events.append(f"faction_join:{faction_id}:{inviter_id}")
        out = float(self.config.team_join_reward)
        out += self._award_join_side_bonuses(
            inviter_id=inviter_id, invitee_id=actor_id, faction_id=faction_id
        )
        self._set_action_cooldown(actor_id, "join_faction")
        return out

    def _adjacent_invite_candidate(self, actor_id: str) -> str | None:
        assert self.state is not None
        actor = self.state.agents[actor_id]
        for other_id in self.possible_agents:
            if other_id == actor_id:
                continue
            other = self.state.agents[other_id]
            if not other.alive:
                continue
            if self._manhattan(actor.position, other.position) != 1:
                continue
            return other_id
        return None

    def _pending_invite_faction_id(self, aid: str) -> int | None:
        assert self.state is not None
        pending = self.state.pending_faction_invites.get(aid)
        if not pending:
            return None
        if self._invite_is_expired(pending):
            self.state.pending_faction_invites.pop(aid, None)
            return None
        return int(pending.get("faction_id", -1))

    def _is_faction_leader(self, aid: str) -> bool:
        assert self.state is not None
        agent = self.state.agents.get(aid)
        if agent is None or agent.faction_id < 0:
            return False
        return self.state.faction_leaders.get(int(agent.faction_id)) == aid

    def _faction_member_count(self, faction_id: int) -> int:
        assert self.state is not None
        if int(faction_id) < 0:
            return 1
        return sum(
            1
            for agent in self.state.agents.values()
            if agent.alive and int(agent.faction_id) == int(faction_id)
        )

    def _is_allied(self, a: AgentState, b: AgentState) -> bool:
        if int(a.faction_id) < 0 or int(b.faction_id) < 0:
            return False
        return int(a.faction_id) == int(b.faction_id)

    def _faction_relation(self, a: AgentState, b: AgentState) -> str:
        if self._is_allied(a, b):
            return "ally"
        if int(a.faction_id) < 0 or int(b.faction_id) < 0:
            return "neutral"
        return "enemy"

    def _queue_deferred_reward(self, aid: str, delta: float, event: str) -> None:
        if delta == 0.0:
            return
        self._deferred_agent_rewards[aid] = self._deferred_agent_rewards.get(aid, 0.0) + float(
            delta
        )
        evts = self._deferred_agent_events.setdefault(aid, [])
        evts.append(str(event))

    def _action_cooldown_blocked(
        self, aid: str, key: str, events: List[str]
    ) -> bool:
        now = int(self.state.step_count) if self.state is not None else 0
        next_allowed = int(self._faction_action_next_step.get(aid, {}).get(key, 0))
        if now < next_allowed:
            events.append(f"{key}_cooldown:{next_allowed - now}")
            return True
        return False

    def _set_action_cooldown(self, aid: str, key: str) -> None:
        now = int(self.state.step_count) if self.state is not None else 0
        cooldown = max(0, int(self.config.faction_action_cooldown_steps))
        if cooldown <= 0:
            return
        bag = self._faction_action_next_step.setdefault(aid, {})
        bag[key] = now + cooldown

    def _pair_token(self, action: str, a: str, b: str) -> str:
        lo, hi = sorted((str(a), str(b)))
        return f"{action}:{lo}|{hi}"

    def _team_pair_reward_allowed(
        self,
        action: str,
        a: str,
        b: str,
        cap: int,
        repeat_guard: int,
        events: List[str],
    ) -> bool:
        token = self._pair_token(action, a, b)
        now = int(self.state.step_count) if self.state is not None else 0
        count = int(self._team_pair_reward_counts.get(token, 0))
        if cap > 0 and count >= cap:
            events.append(f"{action}_reward_guard:cap")
            return False
        if repeat_guard > 0:
            last = self._team_pair_last_reward_step.get(token)
            if last is not None and now - int(last) < int(repeat_guard):
                events.append(f"{action}_reward_guard:repeat")
                return False
        self._team_pair_reward_counts[token] = count + 1
        self._team_pair_last_reward_step[token] = now
        return True

    def _invite_is_expired(self, invite: Dict[str, int | str]) -> bool:
        ttl = int(self.config.faction_invite_ttl_steps)
        if ttl <= 0:
            return True
        created = int(invite.get("created_step", -10**9))
        now = int(self.state.step_count) if self.state is not None else 0
        return (now - created) > ttl

    def _prune_expired_invites(self) -> None:
        if self.state is None:
            return
        for aid, invite in list(self.state.pending_faction_invites.items()):
            if self._invite_is_expired(invite):
                self.state.pending_faction_invites.pop(aid, None)

    def _award_join_side_bonuses(
        self, inviter_id: str, invitee_id: str, faction_id: int
    ) -> float:
        # Anti-exploit rule: invitee can only trigger join-side bonuses once per episode.
        if invitee_id in self._join_bonus_awarded_invitees:
            return 0.0
        self._join_bonus_awarded_invitees.add(invitee_id)
        inviter_bonus = float(self.config.team_join_inviter_reward)
        invitee_bonus = float(self.config.team_join_invitee_reward)
        self._queue_deferred_reward(
            inviter_id,
            inviter_bonus,
            f"faction_join_inviter_bonus:{invitee_id}:{faction_id}:{round(inviter_bonus, 4)}",
        )
        return invitee_bonus

    def _has_adjacent_ally(self, actor: AgentState) -> bool:
        assert self.state is not None
        if int(actor.faction_id) < 0:
            return False
        for other in self.state.agents.values():
            if other.agent_id == actor.agent_id or not other.alive:
                continue
            if not self._is_allied(actor, other):
                continue
            if self._manhattan(actor.position, other.position) == 1:
                return True
        return False

    def _apply_guard_reduction(
        self, target: AgentState, damage: int, events: List[str]
    ) -> int:
        if damage <= 0:
            return damage
        guardian_id = self._guard_assignments.get(target.agent_id)
        if not guardian_id or self.state is None:
            return damage
        guardian = self.state.agents.get(guardian_id)
        if guardian is None or not guardian.alive:
            return damage
        if self._manhattan(guardian.position, target.position) != 1:
            return damage
        reduction = max(
            1,
            int(round(float(damage) * float(self.config.guard_damage_reduction_ratio))),
        )
        reduced = max(0, damage - reduction)
        events.append(
            f"team_guard_block:{guardian_id}:{target.agent_id}:{reduction}:{damage}->{reduced}"
        )
        return reduced

    def _attack(self, actor: AgentState, actor_id: str, events: List[str]) -> float:
        assert self.state is not None
        reward = 0.0
        for monster in self.state.monsters.values():
            if not monster.alive:
                continue
            if (
                abs(monster.position[0] - actor.position[0])
                + abs(monster.position[1] - actor.position[1])
                == 1
            ):
                reward += self._attack_monster(actor, monster, events)
                return reward

        for animal in self.state.animals.values():
            if not animal.alive:
                continue
            if (
                abs(animal.position[0] - actor.position[0])
                + abs(animal.position[1] - actor.position[1])
                == 1
            ):
                reward += self._attack_animal(actor, animal, events)
                return reward

        for other_id, other in self.state.agents.items():
            if other_id == actor_id or not other.alive:
                continue
            if (
                abs(other.position[0] - actor.position[0])
                + abs(other.position[1] - actor.position[1])
                == 1
            ):
                reward += self._attack_agent(actor, other, events)
                return reward
        events.append("attack_no_target")
        reward -= 0.01
        return reward

    def _attack_animal(
        self, attacker: AgentState, target: AnimalState, events: List[str]
    ) -> float:
        weapon, damage_type, damage_range, skill_name = self._equipped_weapon(attacker)
        skill_level = self._skill_level(attacker, skill_name)
        hit_chance = 0.75 + 0.015 * float(attacker.dexterity - 5)
        hit_chance += 0.02 * float(skill_level)
        hit_chance = max(0.25, min(0.95, hit_chance))
        if self._rng.random() > hit_chance:
            events.append(f"agent_interact:attack_animal:{target.animal_id}:{weapon}:{damage_type}")
            events.append(f"agent_interact:miss_animal:{target.entity_id}")
            self._consume_attack_weapon_durability(attacker, weapon, events)
            return -0.01

        raw_damage = self._rng.randint(damage_range[0], damage_range[1])
        raw_damage += self._damage_stat_bonus(attacker, damage_type)
        raw_damage += max(0, (skill_level - 1) // 2)
        raw_damage += int(round(self._agent_enchant_bonus(attacker, "damage_plus")))
        final_damage = max(0, raw_damage)
        events.append(f"agent_interact:attack_animal:{target.animal_id}:{weapon}:{damage_type}")
        events.append(f"agent_interact:attack_roll_animal:{raw_damage}:final:{final_damage}")
        if final_damage <= 0:
            events.append(f"agent_interact:blocked_animal:{target.entity_id}")
            return -0.01

        target.hp = max(0, int(target.hp) - int(final_damage))
        events.append(f"agent_interact:hit_animal:{target.entity_id}")
        reward = 0.04 + 0.015 * float(final_damage)
        if target.hp <= 0 and target.alive:
            target.alive = False
            events.append(f"agent_interact:kill_animal:{target.animal_id}")
            self._drop_animal_material(target, events)
            reward += 0.2
        self._consume_attack_weapon_durability(attacker, weapon, events)
        return reward

    def _attack_agent(
        self, attacker: AgentState, target: AgentState, events: List[str]
    ) -> float:
        allied_target = self._is_allied(attacker, target)
        weapon, damage_type, damage_range, skill_name = self._equipped_weapon(attacker)
        skill_level = self._skill_level(attacker, skill_name)
        hit_chance = self._hit_chance(
            attacker=attacker, target=target, skill_level=skill_level
        )
        if self._rng.random() > hit_chance:
            events.append(
                f"agent_interact:attack:{target.agent_id}:{weapon}:{damage_type}"
            )
            events.append(f"agent_interact:miss:{target.agent_id}")
            self._consume_attack_weapon_durability(attacker, weapon, events)
            return -0.01

        raw_damage = self._rng.randint(damage_range[0], damage_range[1])
        raw_damage += self._damage_stat_bonus(attacker, damage_type)
        raw_damage += max(0, (skill_level - 1) // 2)
        dr, hit_slot, armor_mitigation, armor_skill = self._roll_hit_location_dr(
            target, damage_type
        )
        final_damage = max(0, raw_damage - dr)
        final_damage += int(round(self._agent_enchant_bonus(attacker, "damage_plus")))
        final_damage = self._apply_guard_reduction(target, final_damage, events)

        events.append(
            f"agent_interact:attack:{target.agent_id}:{weapon}:{damage_type}"
        )
        events.append(
            f"agent_interact:attack_roll:{raw_damage}:slot:{hit_slot}:dr:{dr}:final:{final_damage}"
        )
        if armor_skill and armor_mitigation > 0 and final_damage < raw_damage:
            self._gain_skill_xp(
                target,
                armor_skill,
                max(1, armor_mitigation // 2),
                events,
            )

        reward = 0.0
        if final_damage > 0:
            target.hp = max(0, target.hp - final_damage)
            events.append(f"agent_interact:hit:{target.agent_id}")
            if allied_target:
                ff_penalty = float(self.config.ally_damage_penalty_per_hp) * float(
                    final_damage
                )
                reward -= ff_penalty
                events.append(
                    f"ally_damage_penalty:{target.agent_id}:{final_damage}:{round(ff_penalty, 4)}"
                )
            else:
                reward += 0.05 + 0.02 * final_damage
                self._gain_skill_xp(attacker, skill_name, 2, events)
            if target.hp <= 0 and target.alive:
                target.alive = False
                events.append(f"agent_interact:kill:{target.agent_id}")
                if allied_target:
                    reward -= float(self.config.ally_kill_penalty)
                    events.append(f"ally_kill_penalty:{target.agent_id}")
                else:
                    reward += 0.5
                    self._gain_skill_xp(attacker, skill_name, 4, events)
            elif not allied_target and self._rng.random() < 0.18:
                tmp_info = {target.agent_id: {"events": []}}
                tmp_rewards = {target.agent_id: 0.0}
                self._apply_status(
                    target_aid=target.agent_id,
                    status_id="bleed",
                    source_aid=attacker.agent_id,
                    info=tmp_info,
                    rewards=tmp_rewards,
                )
                events.extend(list(tmp_info[target.agent_id]["events"]))
        else:
            events.append(f"agent_interact:blocked:{target.agent_id}")
            reward -= 0.01
        self._consume_attack_weapon_durability(attacker, weapon, events)
        return reward

    def _attack_monster(
        self, attacker: AgentState, target: MonsterState, events: List[str]
    ) -> float:
        weapon, damage_type, damage_range, skill_name = self._equipped_weapon(attacker)
        skill_level = self._skill_level(attacker, skill_name)
        hit_chance = 0.68 + 0.015 * float(attacker.dexterity - 5)
        hit_chance += 0.02 * float(skill_level)
        hit_chance -= 0.02 * float(target.eva)
        hit_chance = max(0.2, min(0.95, hit_chance))

        if self._rng.random() > hit_chance:
            events.append(
                f"agent_interact:attack_monster:{target.monster_id}:{weapon}:{damage_type}"
            )
            events.append(f"agent_interact:miss_monster:{target.entity_id}")
            self._consume_attack_weapon_durability(attacker, weapon, events)
            return -0.01

        raw_damage = self._rng.randint(damage_range[0], damage_range[1])
        raw_damage += self._damage_stat_bonus(attacker, damage_type)
        raw_damage += max(0, (skill_level - 1) // 2)
        dr = self._rng.randint(target.dr_min, max(target.dr_min, target.dr_max))
        final_damage = max(0, raw_damage - dr)
        final_damage += int(round(self._agent_enchant_bonus(attacker, "damage_plus")))

        events.append(
            f"agent_interact:attack_monster:{target.monster_id}:{weapon}:{damage_type}"
        )
        events.append(
            f"agent_interact:attack_roll_monster:{raw_damage}:dr:{dr}:final:{final_damage}"
        )

        reward = 0.0
        if final_damage > 0:
            target.hp = max(0, target.hp - final_damage)
            events.append(f"agent_interact:hit_monster:{target.entity_id}")
            reward += 0.05 + 0.02 * final_damage
            self._gain_skill_xp(attacker, skill_name, 2, events)
            if target.hp <= 0 and target.alive:
                target.alive = False
                events.append(f"agent_interact:kill_monster:{target.monster_id}")
                reward += 0.45
                self._gain_skill_xp(attacker, skill_name, 4, events)
                self._drop_monster_loot(target, events)
        else:
            events.append(f"agent_interact:blocked_monster:{target.entity_id}")
            reward -= 0.01
        self._consume_attack_weapon_durability(attacker, weapon, events)
        return reward

    def _consume_attack_weapon_durability(
        self, attacker: AgentState, weapon_base: str, events: List[str]
    ) -> None:
        if weapon_base == "unarmed":
            return
        for item in reversed(attacker.equipped):
            if self._item_base_id(item) != weapon_base:
                continue
            self._consume_item_durability(attacker, item, 1, events)
            return

    def _equipped_weapon(self, agent: AgentState) -> Tuple[str, str, Tuple[int, int], str]:
        for item in reversed(agent.equipped):
            base = self._item_base_id(item)
            if base in self.weapon_damage_type:
                skill_name = self.weapon_skill_by_item.get(base, "melee")
                return (
                    base,
                    self.weapon_damage_type[base],
                    self.weapon_damage_range[base],
                    skill_name,
                )
        return "unarmed", DAMAGE_TYPE_BLUNT, UNARMED_DAMAGE_RANGE, "melee"

    def _roll_hit_slot(self) -> str:
        slots = [slot for slot, _ in HIT_SLOT_WEIGHTS]
        weights = [weight for _, weight in HIT_SLOT_WEIGHTS]
        return str(self._rng.choices(slots, weights=weights, k=1)[0])

    def _roll_hit_location_dr(
        self,
        target: AgentState,
        damage_type: str,
        forced_hit_slot: str | None = None,
    ) -> Tuple[int, str, int, str]:
        hit_slot = str(forced_hit_slot or self._roll_hit_slot())
        armor_slots = HIT_SLOT_TO_ARMOR_SLOTS.get(hit_slot, ())
        armor_mitigation = 0
        has_armor_in_slot = False
        best_armor_class = ""
        best_armor_score = -1
        for slot in armor_slots:
            item = target.armor_slots.get(slot)
            if item is None:
                continue
            has_armor_in_slot = True
            base = self._item_base_id(item)
            armor_mitigation += int(self.item_dr_bonus_vs.get(base, {}).get(damage_type, 0))
            armor_mitigation += int(round(self._enchant_bonus_for_item(item, "defense_plus")))
            armor_class = self._armor_class_for_item(base)
            score = 0
            if armor_class == "heavy":
                score = 3
            elif armor_class == "medium":
                score = 2
            elif armor_class == "light":
                score = 1
            if score > best_armor_score:
                best_armor_score = score
                best_armor_class = armor_class

        armor_skill = ""
        if has_armor_in_slot:
            armor_skill = ARMOR_CLASS_TO_SKILL.get(best_armor_class, "")
            if armor_skill:
                armor_mitigation += max(0, self._skill_level(target, armor_skill) // 3)

        race = self._race_by_name(target.race_name)
        base_min = int(race.base_dr_min)
        base_max = int(race.base_dr_max)
        if base_max < base_min:
            base_max = base_min
        dr = (base_min + base_max) // 2
        dr += int(race.dr_bonus_vs.get(damage_type, 0))
        dr += armor_mitigation
        if target.agent_id in self._defending_agents:
            defend_bonus, _, _ = self._defend_dr_bonus_for(target.agent_id)
            dr += int(defend_bonus)
        if self._has_adjacent_ally(target):
            dr += int(self.config.formation_dr_bonus)
        return max(0, dr), hit_slot, max(0, armor_mitigation), armor_skill

    def _roll_armor_dr(self, target: AgentState, damage_type: str) -> int:
        dr, _, _, _ = self._roll_hit_location_dr(target, damage_type)
        return dr

    def _armor_class_for_item(self, base_item: str) -> str:
        explicit = str(self.armor_class_by_item.get(base_item, "")).strip().lower()
        if explicit in {"light", "medium", "heavy"}:
            return explicit
        if base_item.startswith("steel_"):
            return "heavy"
        if base_item.startswith("chain_") or base_item.startswith("bronze_"):
            return "medium"
        if base_item in {"chain_mail", "shield"}:
            return "medium"
        if base_item.startswith("leather_"):
            return "light"
        if base_item.startswith("ring_") or base_item in {
            "amulet",
            "silver_necklace",
            "guardian_torc",
        }:
            return "light"
        return "light"

    def _pickup_from_tile(self, agent: AgentState, events: List[str]) -> float:
        assert self.state is not None
        r, c = agent.position
        reward = 0.0

        items = self.state.ground_items.get((r, c), [])
        if items:
            item = items.pop(0)
            added = self._add_item_or_drop(agent, item, 1, events)
            base_item = self._item_base_id(item)
            if added <= 0:
                reward -= 0.01
                events.append("pickup_blocked")
            else:
                reward += 0.1
                events.append(f"pickup:{item}")
            if base_item in ("bow", "crossbow"):
                self._gain_skill_xp(agent, "archery", 1, events)
            elif base_item in ("thrown_rock", "thrown_knife"):
                self._gain_skill_xp(agent, "thrown_weapons", 1, events)
            elif base_item in ("bandage", "healing_potion", "antidote"):
                self._gain_skill_xp(agent, "medic", 1, events)
            return reward

        chest = self.state.chests.get((r, c))
        if chest and not chest.opened:
            chest.opened = True
            if chest.loot:
                for item in chest.loot:
                    self._add_item_or_drop(agent, item, 1, events)
                    events.append(f"chest_open:{item}")
                reward += 0.16 + 0.06 * min(4, len(chest.loot))
            else:
                events.append("chest_open:empty")
                reward -= 0.01
            return reward

        reward -= 0.02
        events.append("loot_fail")

        return reward

    def _apply_survival_costs(
        self,
        agent: AgentState,
        rewards: Dict[str, float],
        aid: str,
        info: Dict[str, Dict[str, object]],
    ) -> float:
        delta = 0.0
        # Passive spell resource updates each step.
        if agent.max_mana > 0 and agent.mana < agent.max_mana:
            agent.mana = min(agent.max_mana, agent.mana + 1)
        for spell_id in list(agent.spell_cooldowns.keys()):
            remain = int(agent.spell_cooldowns.get(spell_id, 0))
            if remain <= 1:
                agent.spell_cooldowns.pop(spell_id, None)
            else:
                agent.spell_cooldowns[spell_id] = remain - 1
        if not self.config.hunger_tick_enabled:
            return delta
        agent.hunger = max(0, agent.hunger - 1)
        if agent.hunger == 0:
            agent.hp -= 1
            rewards[aid] -= 0.05
            delta -= 0.05
            info[aid]["events"].append("starve_tick")
        hunger_ratio = agent.hunger / max(1, agent.max_hunger)
        if hunger_ratio < LOW_HUNGER_THRESHOLD:
            pressure = (LOW_HUNGER_THRESHOLD - hunger_ratio) / LOW_HUNGER_THRESHOLD
            penalty = LOW_HUNGER_PENALTY_SCALE * pressure
            rewards[aid] -= penalty
            delta -= penalty
            info[aid]["events"].append("low_hunger_pressure")
        return delta

    def _apply_search_and_exploration_rewards(
        self,
        aid: str,
        rewards: Dict[str, float],
        info: Dict[str, Dict[str, object]],
        pre_enemy_distance: int | None,
        pre_enemy_visible: bool,
    ) -> float:
        assert self.state is not None
        agent = self.state.agents[aid]
        delta_total = 0.0
        if not agent.alive:
            return delta_total

        metrics = self._episode_metrics.get(aid)
        if metrics is None:
            return delta_total

        seen_tiles = metrics.setdefault("seen_tiles", set())
        if not isinstance(seen_tiles, set):
            seen_tiles = set()
            metrics["seen_tiles"] = seen_tiles

        visible_now = self._visible_tile_coords(aid)
        unseen_before = set(seen_tiles)
        new_tiles = [pos for pos in visible_now if pos not in unseen_before]
        if new_tiles:
            seen_tiles.update(new_tiles)
            metrics["steps_since_new_tile"] = 0
            bonus = float(self.config.new_tile_seen_reward) * float(len(new_tiles))
            rewards[aid] += bonus
            delta_total += bonus
            info[aid]["events"].append(f"new_tiles_seen:{len(new_tiles)}")
        else:
            steps_without = int(metrics.get("steps_since_new_tile", 0)) + 1
            metrics["steps_since_new_tile"] = steps_without
            if steps_without >= max(1, int(self.config.stagnation_threshold_steps)):
                penalty = float(self.config.stagnation_penalty)
                rewards[aid] -= penalty
                delta_total -= penalty
                info[aid]["events"].append("stagnation_penalty")

        action = int(info[aid].get("action", ACTION_WAIT))
        moved = any(
            isinstance(evt, str) and evt.startswith("move:")
            for evt in info[aid]["events"]
        )
        if moved and action in MOVE_DELTAS and self._is_frontier_tile(agent.position, unseen_before):
            bonus = float(self.config.frontier_step_reward)
            rewards[aid] += bonus
            delta_total += bonus
            info[aid]["events"].append("frontier_step")

        curr_enemy_visible = self._enemy_visible(aid)
        curr_enemy_distance = self._nearest_opponent_distance(aid)
        if curr_enemy_visible:
            self._episode_any_enemy_seen = True
            metrics["ever_enemy_seen"] = True
            metrics["enemy_visible_steps"] = int(metrics.get("enemy_visible_steps", 0)) + 1
            if metrics.get("first_enemy_seen_step") is None:
                metrics["first_enemy_seen_step"] = int(self.state.step_count)
                bonus = float(self.config.first_enemy_seen_bonus)
                rewards[aid] += bonus
                delta_total += bonus
                info[aid]["events"].append("first_enemy_seen")
            bonus = float(self.config.enemy_visible_reward)
            rewards[aid] += bonus
            delta_total += bonus
            info[aid]["events"].append("enemy_visible")
        if pre_enemy_visible and not curr_enemy_visible:
            penalty = float(self.config.lost_enemy_penalty)
            rewards[aid] -= penalty
            delta_total -= penalty
            info[aid]["events"].append("enemy_lost")
        if pre_enemy_distance is not None and curr_enemy_distance is not None:
            delta = float(pre_enemy_distance - curr_enemy_distance)
            clip = max(0.0, float(self.config.enemy_distance_delta_clip))
            if clip > 0.0:
                delta = max(-clip, min(clip, delta))
            if abs(delta) > 0.0:
                contrib = float(self.config.enemy_distance_delta_reward_scale) * delta
                rewards[aid] += contrib
                delta_total += contrib
                info[aid]["events"].append(f"enemy_distance_delta:{delta:.2f}")
            metrics["enemy_distance_delta_sum"] = float(
                metrics.get("enemy_distance_delta_sum", 0.0)
            ) + float(delta)
            metrics["enemy_distance_delta_count"] = int(
                metrics.get("enemy_distance_delta_count", 0)
            ) + 1
        metrics["last_enemy_distance"] = curr_enemy_distance

        combat_exchange = any(
            isinstance(evt, str)
            and (
                evt.startswith("agent_interact:attack:")
                or evt.startswith("agent_interact:attack_monster:")
                or evt.startswith("monster_attack:")
                or evt.startswith("monster_hit:")
                or evt.startswith("monster_miss:")
                or evt.startswith("agent_interact:hit:")
                or evt.startswith("agent_interact:hit_monster:")
            )
            for evt in info[aid]["events"]
        )
        if combat_exchange:
            metrics["combat_exchanges"] = int(metrics.get("combat_exchanges", 0)) + 1
            self._episode_combat_exchanges += 1
        return delta_total

    def _apply_team_rewards(
        self,
        rewards: Dict[str, float],
        reward_components: Dict[str, Dict[str, float]],
        info: Dict[str, Dict[str, object]],
    ) -> None:
        assert self.state is not None
        faction_generated: Dict[int, float] = {}
        for aid in self.possible_agents:
            agent = self.state.agents[aid]
            if not agent.alive:
                continue
            generated = 0.0
            for evt in info[aid].get("events", []):
                if not isinstance(evt, str):
                    continue
                if evt.startswith("faction_create:"):
                    generated += float(self.config.team_create_reward)
                elif evt.startswith("faction_invite_sent:"):
                    generated += float(self.config.team_invite_reward)
                elif evt.startswith("faction_join:"):
                    generated += float(self.config.team_join_reward)
                elif evt.startswith("team_give:"):
                    generated += float(self.config.team_give_reward)
                elif evt.startswith("team_trade:"):
                    generated += float(self.config.team_trade_reward)
                elif evt.startswith("team_revive:"):
                    generated += float(self.config.team_revive_reward)
                elif evt.startswith("team_guard:"):
                    generated += float(self.config.team_guard_reward)
                elif evt.startswith("faction_leave:"):
                    generated += float(self.config.team_leave_reward)

            if int(agent.faction_id) >= 0:
                has_adjacent_ally = False
                proximity_partner: str | None = None
                for other_id, other in self.state.agents.items():
                    if other_id == aid or not other.alive:
                        continue
                    if not self._is_allied(agent, other):
                        continue
                    if self._manhattan(agent.position, other.position) == 1:
                        has_adjacent_ally = True
                        proximity_partner = other_id
                        break
                if has_adjacent_ally:
                    allowed = True
                    if proximity_partner is not None:
                        allowed = self._team_pair_reward_allowed(
                            action="team_proximity",
                            a=aid,
                            b=proximity_partner,
                            cap=max(
                                0, int(self.config.team_proximity_pair_cap_per_episode)
                            ),
                            repeat_guard=max(
                                0,
                                int(self.config.team_proximity_repeat_guard_steps),
                            ),
                            events=info[aid]["events"],
                        )
                    if allowed:
                        prox = float(self.config.team_proximity_reward)
                        rewards[aid] += prox
                        reward_components[aid]["teamwork"] += prox
                        generated += prox
                        info[aid]["events"].append("team_proximity")

            if int(agent.faction_id) >= 0 and generated > 0.0:
                faction_generated[int(agent.faction_id)] = faction_generated.get(
                    int(agent.faction_id), 0.0
                ) + generated

        for faction_id, generated in faction_generated.items():
            leader_id = self.state.faction_leaders.get(int(faction_id))
            if not leader_id:
                continue
            leader = self.state.agents.get(leader_id)
            if leader is None or not leader.alive:
                continue
            share = float(self.config.leader_team_share) * float(generated)
            if share <= 0.0:
                continue
            rewards[leader_id] += share
            reward_components[leader_id]["teamwork"] += share
            info[leader_id]["events"].append(
                f"leader_team_share:{faction_id}:{round(share, 4)}"
            )

    def _count_held_treasures(self, agent: AgentState) -> int:
        count = 0
        for item in agent.inventory + agent.equipped:
            if self._item_base_id(item) in self.treasure_items:
                count += 1
        return count

    def _apply_treasure_rewards(
        self,
        rewards: Dict[str, float],
        reward_components: Dict[str, Dict[str, float]],
        info: Dict[str, Dict[str, object]],
    ) -> None:
        if not self.treasure_items:
            return
        cap = max(0, int(self.config.treasure_hold_reward_cap_items))
        per_turn = float(self.config.treasure_hold_reward_per_turn)
        if per_turn == 0.0:
            return
        for aid in self.possible_agents:
            if self.state is None:
                break
            agent = self.state.agents[aid]
            if not agent.alive:
                continue
            held = self._count_held_treasures(agent)
            if held <= 0:
                continue
            rewarded_count = held if cap <= 0 else min(held, cap)
            delta = float(rewarded_count) * per_turn
            rewards[aid] += delta
            reward_components[aid]["treasure"] += delta
            info[aid]["events"].append(f"treasure_hold:{held}")

    def _apply_treasure_end_bonus(
        self,
        rewards: Dict[str, float],
        reward_components: Dict[str, Dict[str, float]],
        info: Dict[str, Dict[str, object]],
    ) -> None:
        if not self.treasure_items:
            return
        per_item = float(self.config.treasure_end_bonus_per_item)
        if per_item <= 0.0:
            return
        for aid in self.possible_agents:
            if self.state is None:
                break
            agent = self.state.agents[aid]
            if not agent.alive:
                continue
            held = self._count_held_treasures(agent)
            if held <= 0:
                continue
            delta = per_item * float(held)
            rewards[aid] += delta
            reward_components[aid]["treasure"] += delta
            info[aid]["events"].append(f"treasure_end_bonus:{held}")

    def _apply_focus_rewards(
        self,
        rewards: Dict[str, float],
        reward_components: Dict[str, Dict[str, float]],
        info: Dict[str, Dict[str, object]],
    ) -> None:
        for aid in self.possible_agents:
            treasure_events = 0
            combat_events = 0
            skill_events = 0
            team_events = 0
            for evt in info[aid].get("events", []):
                if not isinstance(evt, str):
                    continue
                if (
                    evt.startswith("pickup:")
                    or evt.startswith("loot:")
                    or evt.startswith("chest_open:")
                    or evt.startswith("monster_loot_drop:")
                ):
                    treasure_events += 1
                if (
                    evt.startswith("agent_interact:hit:")
                    or evt.startswith("agent_interact:kill:")
                    or evt.startswith("agent_interact:hit_monster:")
                    or evt.startswith("agent_interact:kill_monster:")
                    or evt.startswith("monster_hit:")
                ):
                    combat_events += 1
                if evt.startswith("skill_up:"):
                    skill_events += 1
                if evt.startswith("faction_") or evt.startswith("team_"):
                    team_events += 1

            focus_delta = 0.0
            focus_delta += 0.02 * float(treasure_events)
            focus_delta += 0.02 * float(combat_events)
            focus_delta += 0.03 * float(skill_events)
            focus_delta += float(self.config.team_action_bonus) * float(team_events)
            if focus_delta == 0.0:
                continue
            rewards[aid] += focus_delta
            reward_components[aid]["focus"] += focus_delta

    def _observation_window_dims(self, aid: str) -> Tuple[int, int]:
        assert self.state is not None
        cfg = self.config.agent_observation_config.get(aid, {})
        agent = self.state.agents[aid]
        profile = self._profile_for_agent(aid)
        exploration_bonus = max(0, self._skill_level(agent, "exploration"))
        base_vision = max(1, int(self.config.vision_range_default))
        view_width = max(base_vision, int(cfg.get("view_width", profile.view_width))) + exploration_bonus
        view_height = max(base_vision, int(cfg.get("view_height", profile.view_height))) + exploration_bonus
        return max(1, view_height), max(1, view_width)

    def _tile_blocks_los(self, tile_id: str) -> bool:
        if str(tile_id) in OPAQUE_TILE_IDS:
            return True
        if tile_id in self.tiles:
            return not bool(self.tiles[tile_id].walkable)
        return False

    def _line_points(self, a: Tuple[int, int], b: Tuple[int, int]) -> List[Tuple[int, int]]:
        x0, y0 = int(a[0]), int(a[1])
        x1, y1 = int(b[0]), int(b[1])
        points: List[Tuple[int, int]] = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        while True:
            points.append((x0, y0))
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x0 += sx
            if e2 < dx:
                err += dx
                y0 += sy
        return points

    def _has_line_of_sight(self, src: Tuple[int, int], dst: Tuple[int, int]) -> bool:
        assert self.state is not None
        if src == dst:
            return True
        if not bool(self.config.enable_los):
            return True
        ray = self._line_points(src, dst)
        for rr, cc in ray[1:-1]:
            if rr < 0 or cc < 0 or rr >= len(self.state.grid) or cc >= len(self.state.grid[0]):
                return False
            if self._tile_blocks_los(self.state.grid[rr][cc]):
                return False
        return True

    def _visible_tile_coords(self, aid: str) -> List[Tuple[int, int]]:
        assert self.state is not None
        if aid not in self.state.agents:
            return []
        agent = self.state.agents[aid]
        if not agent.alive:
            return []
        view_height, view_width = self._observation_window_dims(aid)
        cr, cc = agent.position
        start_r = cr - (view_height // 2)
        start_c = cc - (view_width // 2)
        out: List[Tuple[int, int]] = []
        for r in range(start_r, start_r + view_height):
            for c in range(start_c, start_c + view_width):
                if (
                    r < 0
                    or c < 0
                    or r >= len(self.state.grid)
                    or c >= len(self.state.grid[0])
                ):
                    continue
                if self._has_line_of_sight(agent.position, (r, c)):
                    out.append((r, c))
        return out

    def _is_frontier_tile(self, pos: Tuple[int, int], seen_before: set[Tuple[int, int]]) -> bool:
        assert self.state is not None
        r, c = pos
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (
                nr < 0
                or nc < 0
                or nr >= len(self.state.grid)
                or nc >= len(self.state.grid[0])
            ):
                continue
            if (nr, nc) not in seen_before:
                return True
        return False

    def _nearest_opponent_distance(self, aid: str) -> int | None:
        assert self.state is not None
        actor = self.state.agents[aid]
        if not actor.alive:
            return None
        best: int | None = None
        for other_id, other in self.state.agents.items():
            if other_id == aid or not other.alive:
                continue
            if self._is_allied(actor, other):
                continue
            d = self._manhattan(actor.position, other.position)
            if best is None or d < best:
                best = d
        return best

    def _enemy_visible(self, aid: str) -> bool:
        assert self.state is not None
        actor = self.state.agents[aid]
        if not actor.alive:
            return False
        view_height, view_width = self._observation_window_dims(aid)
        half_h = view_height // 2
        half_w = view_width // 2
        ar, ac = actor.position
        for other_id, other in self.state.agents.items():
            if other_id == aid or not other.alive:
                continue
            if self._is_allied(actor, other):
                continue
            dr = abs(other.position[0] - ar)
            dc = abs(other.position[1] - ac)
            if dr <= half_h and dc <= half_w and self._has_line_of_sight(actor.position, other.position):
                return True
        return False

    def _walkable(self, r: int, c: int) -> bool:
        assert self.state is not None
        if r < 0 or c < 0 or r >= len(self.state.grid) or c >= len(self.state.grid[0]):
            return False
        tile_id = self.state.grid[r][c]
        if not self.tiles[tile_id].walkable:
            return False
        for agent in self.state.agents.values():
            if agent.alive and agent.position == (r, c):
                return False
        for monster in self.state.monsters.values():
            if monster.alive and monster.position == (r, c):
                return False
        for animal in self.state.animals.values():
            if animal.alive and animal.position == (r, c):
                return False
        return True

    def _walkable_for_monster(
        self, r: int, c: int, moving_entity_id: str
    ) -> bool:
        assert self.state is not None
        if r < 0 or c < 0 or r >= len(self.state.grid) or c >= len(self.state.grid[0]):
            return False
        tile_id = self.state.grid[r][c]
        if not self.tiles[tile_id].walkable:
            return False
        for agent in self.state.agents.values():
            if agent.alive and agent.position == (r, c):
                return False
        for monster in self.state.monsters.values():
            if (
                monster.alive
                and monster.entity_id != moving_entity_id
                and monster.position == (r, c)
            ):
                return False
        for animal in self.state.animals.values():
            if animal.alive and animal.position == (r, c):
                return False
        return True

    def _build_observation(self, aid: str) -> Dict[str, object]:
        assert self.state is not None
        cfg = self.config.agent_observation_config.get(aid, {})

        agent = self.state.agents[aid]
        profile = self._profile_for_agent(aid)
        view_height, view_width = self._observation_window_dims(aid)
        include_grid = bool(cfg.get("include_grid", profile.include_grid))
        include_stats = bool(cfg.get("include_stats", profile.include_stats))
        include_inventory = bool(
            cfg.get("include_inventory", profile.include_inventory)
        )
        obs: Dict[str, object] = {"step": self.state.step_count, "alive": agent.alive}
        obs["profile"] = profile.name
        obs["race"] = agent.race_name
        obs["class"] = agent.class_name
        obs["faction"] = {
            "faction_id": int(agent.faction_id),
            "is_leader": bool(self._is_faction_leader(aid)),
            "pending_invite_from_faction": self._pending_invite_faction_id(aid),
        }

        if include_grid:
            obs["local_tiles"] = self._local_view_dims(
                agent.position, height=view_height, width=view_width, aid=aid
            )
        if include_stats:
            nearby_item_counts = self._nearby_item_counts(
                center=agent.position, height=view_height, width=view_width
            )
            metrics = self._episode_metrics.get(aid, {})
            seen_tiles = int(len(metrics.get("seen_tiles", set())))
            dcnt = int(metrics.get("enemy_distance_delta_count", 0))
            dsum = float(metrics.get("enemy_distance_delta_sum", 0.0))
            obs["stats"] = {
                "hp": agent.hp,
                "mana": agent.mana,
                "max_mana": agent.max_mana,
                "hunger": agent.hunger,
                "position": agent.position,
                "equipped_count": len(agent.equipped),
                "carried_weight": self._carried_weight(agent),
                "carry_capacity": self._carry_capacity(agent),
                "armor_slots": dict(agent.armor_slots),
                "strength": agent.strength,
                "dexterity": agent.dexterity,
                "intellect": agent.intellect,
                "skills": dict(agent.skills),
                "skill_xp": dict(agent.skill_xp),
                "known_spells": list(agent.known_spells),
                "spell_cooldowns": dict(agent.spell_cooldowns),
                "statuses": [
                    {
                        "id": s.status_id,
                        "remaining": int(s.remaining),
                        "tick_interval": int(s.tick_interval),
                    }
                    for s in self.state.agent_statuses.get(aid, [])
                ],
                "overall_level": self._overall_level(agent),
                "encumbrance_ratio": self._encumbrance_ratio(agent),
                "faction_id": int(agent.faction_id),
                "is_faction_leader": bool(self._is_faction_leader(aid)),
                "faction_member_count": int(self._faction_member_count(agent.faction_id)),
                "nearby_item_counts": nearby_item_counts,
                "tile_interaction_counts": self._tile_interaction_counts(
                    agent.position
                ),
                "teammate_distance": self._nearest_teammate_distance(aid),
                "enemy_distance": self._nearest_opponent_distance(aid),
                "enemy_visible": self._enemy_visible(aid),
                "explore_coverage": float(seen_tiles) / float(self._walkable_tile_count),
                "steps_since_new_tile": int(metrics.get("steps_since_new_tile", 0)),
                "first_enemy_seen_step": metrics.get("first_enemy_seen_step"),
                "enemy_visible_steps": int(metrics.get("enemy_visible_steps", 0)),
                "enemy_distance_delta_mean": (dsum / dcnt) if dcnt > 0 else 0.0,
                "nearby_chests": self._nearby_chest_counts(
                    center=agent.position, height=view_height, width=view_width
                ),
                "nearby_resource_nodes": self._nearby_resource_counts(
                    center=agent.position, height=view_height, width=view_width
                ),
                "nearby_stations": self._nearby_station_counts(
                    center=agent.position, height=view_height, width=view_width
                ),
                "biome": self.state.biomes.get(agent.position, ""),
                "nearby_agents": self._nearby_agent_counts(
                    aid=aid, center=agent.position, height=view_height, width=view_width
                ),
                "nearby_monsters": self._nearby_monster_counts(
                    center=agent.position, height=view_height, width=view_width
                ),
                "nearby_animals": self._nearby_animal_counts(
                    center=agent.position, height=view_height, width=view_width
                ),
            }
        if include_inventory:
            obs["inventory"] = list(agent.inventory)

        return obs

    def _resolve_profile_name(self, agent_id: str, index: int) -> str:
        if agent_id in self.config.agent_profile_map:
            raw = self.config.agent_profile_map[agent_id]
            return PROFILE_ALIASES.get(raw, raw)
        race = self.config.agent_race_map.get(agent_id, "")
        if race:
            alias = PROFILE_ALIASES.get(race, race)
            if alias in self.profiles:
                return alias
        if "reward_explorer_policy_v1" in self.profiles:
            return "reward_explorer_policy_v1"
        ordered = sorted(self.profiles.keys())
        if not ordered:
            raise ValueError("No agent profiles are loaded")
        return ordered[index % len(ordered)]

    def _resolve_class_name(self, agent_id: str, index: int) -> str:
        if agent_id in self.config.agent_class_map:
            return self.config.agent_class_map[agent_id]
        if "fighter" in self.classes:
            return "fighter"
        ordered = sorted(self.classes.keys())
        if not ordered:
            raise ValueError("No agent classes are loaded")
        return ordered[index % len(ordered)]

    def _default_known_spells_for_class(self, class_name: str) -> List[str]:
        base = ["arc_bolt", "cleanse", "haste"]
        if class_name == "rogue":
            base = ["arc_bolt", "blink", "reveal"]
        elif class_name == "fighter":
            base = ["arc_bolt", "cleanse", "haste"]
        elif class_name == "medic":
            base = ["cleanse", "regen_touch", "reveal"]
        return [s for s in base if s in self.spell_defs]

    def _resolve_race_name(self, agent_id: str, index: int) -> str:
        if agent_id in self.config.agent_race_map:
            return self.config.agent_race_map[agent_id]
        if agent_id in self.config.agent_profile_map:
            mapped = self.config.agent_profile_map[agent_id]
            if mapped in self.races:
                return mapped
        if "human" in self.races:
            return "human"
        ordered = sorted(self.races.keys())
        if not ordered:
            raise ValueError("No agent races are loaded")
        return ordered[index % len(ordered)]

    def _profile_by_name(self, name: str) -> AgentProfile:
        resolved = PROFILE_ALIASES.get(name, name)
        if resolved not in self.profiles:
            raise ValueError(f"Unknown agent profile: {name}")
        return self.profiles[resolved]

    def _profile_for_agent(self, agent_id: str) -> AgentProfile:
        assert self.state is not None
        return self._profile_by_name(self.state.agents[agent_id].profile_name)

    def _class_by_name(self, name: str) -> AgentClass:
        if name not in self.classes:
            raise ValueError(f"Unknown agent class: {name}")
        return self.classes[name]

    def _race_by_name(self, name: str) -> AgentRace:
        if name not in self.races:
            raise ValueError(f"Unknown agent race: {name}")
        return self.races[name]

    def _assert_item_known(self, item_id: str, source: str) -> None:
        if item_id not in self.items.items:
            raise ValueError(f"{source} references unknown item '{item_id}'")

    def _validate_item_references(self) -> None:
        for class_name, cls in sorted(self.classes.items()):
            for idx, item in enumerate(cls.starting_items):
                self._assert_item_known(item, f"class '{class_name}' starting_items[{idx}]")
        for tile_id, tile in sorted(self.tiles.items()):
            for idx, item in enumerate(tile.loot_table):
                self._assert_item_known(item, f"tile '{tile_id}' loot_table[{idx}]")
        for monster_id, monster in sorted(self.monsters.items()):
            for idx, loot in enumerate(monster.loot):
                self._assert_item_known(
                    loot.item, f"monster '{monster_id}' loot[{idx}]"
                )
        for animal_id, animal in sorted(self.animals.items()):
            if animal.drop_item:
                self._assert_item_known(
                    animal.drop_item, f"animal '{animal_id}' drop_item"
                )
            if animal.can_shear and animal.shear_item:
                self._assert_item_known(
                    animal.shear_item, f"animal '{animal_id}' shear_item"
                )
        for crop_id, row in sorted(PLANT_TYPES.items()):
            seed_item = str(row.get("seed_item", "")).strip()
            food_item = str(row.get("food_item", "")).strip()
            crop_tile = str(row.get("crop_tile", "")).strip()
            if seed_item:
                self._assert_item_known(seed_item, f"plant '{crop_id}' seed_item")
            if food_item:
                self._assert_item_known(food_item, f"plant '{crop_id}' food_item")
            if crop_tile and crop_tile not in self.tiles:
                raise ValueError(f"plant '{crop_id}' crop_tile '{crop_tile}' is unknown")

    def _validate_static_map_references(self) -> None:
        if self.static_map_layout is None:
            return
        for r, row in enumerate(self.static_map_layout.grid):
            for c, tile_id in enumerate(row):
                if tile_id not in self.tiles:
                    raise ValueError(
                        f"static map grid references unknown tile '{tile_id}' at ({r}, {c})"
                    )

    def _validate_recipe_references(self) -> None:
        for recipe_id, recipe in sorted(self.recipes.items()):
            for idx, item_id in enumerate(sorted(recipe.inputs.keys())):
                self._assert_item_known(
                    item_id, f"recipe '{recipe_id}' inputs[{idx}]"
                )
            for idx, item_id in enumerate(sorted(recipe.outputs.keys())):
                self._assert_item_known(
                    item_id, f"recipe '{recipe_id}' outputs[{idx}]"
                )
            if recipe.build_tile_id and recipe.build_tile_id not in self.tiles:
                raise ValueError(
                    f"recipe '{recipe_id}' build_tile_id '{recipe.build_tile_id}' is unknown"
                )
            if recipe.required_tool_category:
                categories = set(self.tool_category_by_item.values())
                if recipe.required_tool_category not in categories:
                    raise ValueError(
                        f"recipe '{recipe_id}' required_tool_category '{recipe.required_tool_category}' has no matching items"
                    )
        for idx, row in enumerate(self.mapgen_cfg.resource_nodes):
            if not isinstance(row, dict):
                continue
            drop_item = str(row.get("drop_item", "")).strip()
            if drop_item:
                self._assert_item_known(
                    drop_item, f"mapgen.resource_nodes[{idx}].drop_item"
                )

    def _validate_effect_references(self) -> None:
        for spell_id, spell in sorted(self.spell_defs.items()):
            for idx, reagent in enumerate(sorted(spell.required_reagents.keys())):
                self._assert_item_known(
                    reagent, f"spell '{spell_id}' required_reagents[{idx}]"
                )
            for eff in spell.effects:
                self._validate_effect_item_refs(
                    eff, f"spell '{spell_id}' effect"
                )
        for status_id, status in sorted(self.status_defs.items()):
            for eff in status.apply_effects + status.tick_effects + status.expire_effects:
                self._validate_effect_item_refs(
                    eff, f"status '{status_id}' effect"
                )

    def _validate_effect_item_refs(self, eff: Dict[str, object], source: str) -> None:
        et = str(eff.get("type", "")).strip()
        if et in {"consume_item", "require_item", "grant_item"}:
            item_id = str(eff.get("item", "")).strip()
            if item_id:
                self._assert_item_known(item_id, source)

    def _local_view_dims(
        self, center: Tuple[int, int], height: int, width: int, aid: str = ""
    ) -> List[List[str]]:
        assert self.state is not None
        height = max(1, int(height))
        width = max(1, int(width))
        visible = set(self._visible_tile_coords(aid)) if aid else None
        cr, cc = center
        start_r = cr - (height // 2)
        start_c = cc - (width // 2)
        view: List[List[str]] = []
        for r in range(start_r, start_r + height):
            row: List[str] = []
            for c in range(start_c, start_c + width):
                if (
                    r < 0
                    or c < 0
                    or r >= len(self.state.grid)
                    or c >= len(self.state.grid[0])
                ):
                    row.append("void")
                else:
                    tile_id = self.state.grid[r][c]
                    if visible is not None and (r, c) not in visible:
                        row.append("fog")
                    else:
                        row.append(tile_id)
            view.append(row)
        return view

    def _nearby_item_counts(
        self, center: Tuple[int, int], height: int, width: int
    ) -> Dict[str, int]:
        assert self.state is not None
        counts: Dict[str, int] = {}
        cr, cc = center
        start_r = cr - (height // 2)
        start_c = cc - (width // 2)
        for r in range(start_r, start_r + height):
            for c in range(start_c, start_c + width):
                if (
                    r < 0
                    or c < 0
                    or r >= len(self.state.grid)
                    or c >= len(self.state.grid[0])
                ):
                    continue
                for item in self.state.ground_items.get((r, c), []):
                    counts[item] = counts.get(item, 0) + 1
        return counts

    def _tile_interaction_counts(self, center: Tuple[int, int]) -> Dict[str, int]:
        assert self.state is not None
        used_here = self.state.tile_interactions.get(center, 0)
        used_total = sum(self.state.tile_interactions.values())
        return {"current_tile_used": used_here, "total_used": used_total}

    def _nearest_teammate_distance(self, aid: str) -> int | None:
        assert self.state is not None
        actor = self.state.agents[aid]
        if int(actor.faction_id) < 0:
            return None
        distances: List[int] = []
        for other_id, other in self.state.agents.items():
            if other_id == aid or not other.alive:
                continue
            if not self._is_allied(actor, other):
                continue
            d = abs(actor.position[0] - other.position[0]) + abs(
                actor.position[1] - other.position[1]
            )
            distances.append(d)
        if not distances:
            return None
        return min(distances)

    def _nearest_food_distance(self, position: Tuple[int, int]) -> int | None:
        assert self.state is not None
        row, col = position
        best: int | None = None
        for (r, c), items in self.state.ground_items.items():
            if any(item in self.edible_items for item in items):
                dist = abs(row - r) + abs(col - c)
                if best is None or dist < best:
                    best = dist
        for (r, c), chest in self.state.chests.items():
            if chest.opened:
                continue
            if any(item in self.edible_items for item in chest.loot):
                dist = abs(row - r) + abs(col - c)
                if best is None or dist < best:
                    best = dist

        for r, grid_row in enumerate(self.state.grid):
            for c, tile_id in enumerate(grid_row):
                tile = self.tiles[tile_id]
                if not tile.loot_table:
                    continue
                if not any(item in self.edible_items for item in tile.loot_table):
                    continue
                used = self.state.tile_interactions.get((r, c), 0)
                if used >= max(1, tile.max_interactions):
                    continue
                dist = abs(row - r) + abs(col - c)
                if best is None or dist < best:
                    best = dist
        return best

    def _generate_world_terrain(self) -> Tuple[List[List[str]], Dict[Tuple[int, int], str]]:
        biome_defs = [dict(x) for x in self.mapgen_cfg.biomes if isinstance(x, dict)]
        if not biome_defs:
            grid = generate_map(
                self.config.width,
                self.config.height,
                self.tiles,
                self._rng,
                wall_tile_id=self.mapgen_cfg.wall_tile_id,
                floor_fallback_id=self.mapgen_cfg.floor_fallback_id,
                min_width=self.mapgen_cfg.min_width,
                min_height=self.mapgen_cfg.min_height,
            )
            return grid, {}

        return generate_biome_terrain(
            width=int(self.config.width),
            height=int(self.config.height),
            tiles=self.tiles,
            rng=self._rng,
            biome_defs=biome_defs,
            wall_tile_id=self.mapgen_cfg.wall_tile_id,
            floor_fallback_id=self.mapgen_cfg.floor_fallback_id,
            worldgen=self.mapgen_cfg.worldgen,
            structures_defs=self.structure_defs if self.structure_defs else self.mapgen_cfg.structures,
            min_width=self.mapgen_cfg.min_width,
            min_height=self.mapgen_cfg.min_height,
        )

    def _spawn_resource_nodes(
        self, occupied: List[Tuple[int, int]]
    ) -> Dict[Tuple[int, int], ResourceNodeState]:
        assert self.state is not None
        defs = [dict(x) for x in self.mapgen_cfg.resource_nodes if isinstance(x, dict)]
        if not defs:
            return {}
        occupied_set = set(occupied)
        walkable = [
            (r, c)
            for r, row in enumerate(self.state.grid)
            for c, tile_id in enumerate(row)
            if self.tiles[tile_id].walkable and (r, c) not in occupied_set
        ]
        if not walkable:
            return {}
        self._rng.shuffle(walkable)
        out: Dict[Tuple[int, int], ResourceNodeState] = {}
        for row in defs:
            node_id = str(row.get("id", "")).strip()
            drop_item = str(row.get("drop_item", "")).strip()
            skill = str(row.get("skill", "")).strip()
            if not node_id or not drop_item or not skill:
                continue
            density = max(0.0, float(row.get("density", 0.0)))
            if density <= 0.0:
                continue
            target = max(0, int(len(walkable) * density))
            if target <= 0:
                continue
            min_yield = max(1, int(row.get("min_yield", 1)))
            max_yield = max(min_yield, int(row.get("max_yield", min_yield)))
            allowed_biomes = {
                str(x).strip()
                for x in list(row.get("biomes", []))
                if str(x).strip()
            }
            placed = 0
            for pos in walkable:
                if placed >= target:
                    break
                if pos in out:
                    continue
                biome = str(self.state.biomes.get(pos, ""))
                if allowed_biomes and biome and biome not in allowed_biomes:
                    continue
                qty = self._rng.randint(min_yield, max_yield)
                out[pos] = ResourceNodeState(
                    node_id=node_id,
                    position=pos,
                    skill=skill,
                    drop_item=drop_item,
                    remaining=qty,
                    max_yield=qty,
                    biome=biome,
                )
                placed += 1
        return out

    def _spawn_stations(
        self, occupied: List[Tuple[int, int]]
    ) -> Dict[Tuple[int, int], StationState]:
        assert self.state is not None
        defs = [dict(x) for x in self.mapgen_cfg.station_spawns if isinstance(x, dict)]
        if not defs:
            return {}
        occupied_set = set(occupied)
        walkable = [
            (r, c)
            for r, row in enumerate(self.state.grid)
            for c, tile_id in enumerate(row)
            if self.tiles[tile_id].walkable and (r, c) not in occupied_set
        ]
        if not walkable:
            return {}
        self._rng.shuffle(walkable)
        out: Dict[Tuple[int, int], StationState] = {}
        for row in defs:
            station_id = str(row.get("id", "")).strip()
            if not station_id:
                continue
            density = max(0.0, float(row.get("density", 0.0)))
            if density <= 0.0:
                continue
            target = max(0, int(len(walkable) * density))
            if target <= 0:
                continue
            speed = max(0.1, float(row.get("speed_multiplier", 1.0)))
            quality = max(0, int(row.get("quality_tier", 0)))
            unlock = [str(x).strip() for x in list(row.get("unlock_recipes", [])) if str(x).strip()]
            placed = 0
            for pos in walkable:
                if placed >= target:
                    break
                if pos in out:
                    continue
                out[pos] = StationState(
                    station_id=station_id,
                    position=pos,
                    speed_multiplier=speed,
                    quality_tier=quality,
                    unlock_recipes=unlock,
                )
                placed += 1
        return out

    def _spawn_chests(self, occupied: List[Tuple[int, int]]) -> Dict[Tuple[int, int], ChestState]:
        assert self.state is not None
        occupied_set = set(occupied)
        walkable = [
            (r, c)
            for r, row in enumerate(self.state.grid)
            for c, tile_id in enumerate(row)
            if self.tiles[tile_id].walkable and (r, c) not in occupied_set
        ]
        if not walkable:
            return {}
        n = max(1, int(len(walkable) * float(self.mapgen_cfg.chest_density)))
        self._rng.shuffle(walkable)
        out: Dict[Tuple[int, int], ChestState] = {}
        for pos in walkable[:n]:
            loot_count = self._rng.randint(1, 3)
            loot = [self._rng.choice(self.chest_loot_table) for _ in range(loot_count)]
            out[pos] = ChestState(position=pos, opened=False, locked=False, loot=loot)
        return out

    def _spawn_monsters(
        self, occupied: List[Tuple[int, int]]
    ) -> Dict[str, MonsterState]:
        assert self.state is not None
        density = max(0.0, float(self.mapgen_cfg.monster_density))
        if density <= 0.0:
            return {}

        occupied_set = set(occupied)
        walkable = [
            (r, c)
            for r, row in enumerate(self.state.grid)
            for c, tile_id in enumerate(row)
            if self.tiles[tile_id].walkable and (r, c) not in occupied_set
        ]
        if not walkable:
            return {}

        n_monsters = int(len(walkable) * density)
        if n_monsters <= 0:
            n_monsters = 1
        n_monsters = min(n_monsters, len(walkable))

        spawn_ids = [entry.monster_id for entry in self.monster_spawns]
        spawn_weights = [max(0.0, float(entry.weight)) for entry in self.monster_spawns]
        if not spawn_ids or sum(spawn_weights) <= 0:
            return {}

        self._rng.shuffle(walkable)
        out: Dict[str, MonsterState] = {}
        for idx, pos in enumerate(walkable[:n_monsters]):
            monster_id = self._rng.choices(spawn_ids, weights=spawn_weights, k=1)[0]
            mdef = self.monsters[monster_id]
            entity_id = f"monster_{idx}"
            out[entity_id] = MonsterState(
                entity_id=entity_id,
                monster_id=mdef.monster_id,
                name=mdef.name,
                symbol=mdef.symbol,
                color=mdef.color,
                position=pos,
                hp=mdef.hp,
                max_hp=mdef.hp,
                acc=mdef.acc,
                eva=mdef.eva,
                dmg_min=mdef.dmg_min,
                dmg_max=mdef.dmg_max,
                dr_min=mdef.dr_min,
                dr_max=mdef.dr_max,
                alive=True,
            )
        return out

    def _spawn_animals(
        self, occupied: List[Tuple[int, int]]
    ) -> Dict[str, AnimalState]:
        assert self.state is not None
        density = max(0.0, float(getattr(self.mapgen_cfg, "animal_density", 0.0) or 0.0))
        if density <= 0.0 or not self.animals:
            return {}
        occupied_set = set(occupied)
        walkable = [
            (r, c)
            for r, row in enumerate(self.state.grid)
            for c, tile_id in enumerate(row)
            if self.tiles[tile_id].walkable and (r, c) not in occupied_set
        ]
        if not walkable:
            return {}
        n_animals = max(1, int(len(walkable) * density))
        n_animals = min(n_animals, len(walkable))
        ids = [a.animal_id for a in self.animals.values()]
        weights = [max(0.0, float(a.spawn_weight)) for a in self.animals.values()]
        if sum(weights) <= 0.0:
            return {}
        self._rng.shuffle(walkable)
        out: Dict[str, AnimalState] = {}
        for idx, pos in enumerate(walkable[:n_animals]):
            aid = str(self._rng.choices(ids, weights=weights, k=1)[0])
            adef = self.animals[aid]
            entity_id = f"animal_{idx}"
            out[entity_id] = AnimalState(
                entity_id=entity_id,
                animal_id=adef.animal_id,
                name=adef.name,
                symbol=adef.symbol,
                color=adef.color,
                position=pos,
                hp=adef.hp,
                max_hp=adef.hp,
                hunger=max(2, int(adef.max_hunger // 2)),
                max_hunger=max(3, int(adef.max_hunger)),
                thirst=max(2, int(adef.max_thirst // 2)),
                max_thirst=max(3, int(adef.max_thirst)),
                age=self._rng.randint(0, max(1, adef.mature_age)),
                mature_age=adef.mature_age,
                reproduction_cooldown=self._rng.randint(0, max(1, adef.reproduction_cooldown)),
                reproduction_cooldown_max=adef.reproduction_cooldown,
                prey_score=adef.prey_score,
                movement_speed=adef.movement_speed,
                carnivore=adef.carnivore,
                gender=str(self._rng.choice(["female", "male"])),
                litter_size_min=adef.litter_size_min,
                litter_size_max=adef.litter_size_max,
                can_shear=adef.can_shear,
                sheared=False,
                shear_item=adef.shear_item,
                wool_regrow=0,
                shear_regrow_max=adef.shear_regrow_steps,
                alive=True,
            )
        return out

    def _apply_class_modifiers(self, agent: AgentState, cls: AgentClass) -> None:
        for skill, delta in cls.skill_modifiers.items():
            base = int(agent.skills.get(skill, 0))
            agent.skills[skill] = max(0, base + int(delta))

    def _skill_level(self, agent: AgentState, skill: str) -> int:
        return max(0, int(agent.skills.get(skill, 0)))

    def _gain_skill_xp(
        self, agent: AgentState, skill: str, amount: int, events: List[str]
    ) -> None:
        if amount <= 0:
            return
        current = int(agent.skill_xp.get(skill, 0)) + int(amount)
        level = self._skill_level(agent, skill)
        while current >= self._skill_xp_to_next(level):
            current -= self._skill_xp_to_next(level)
            level += 1
            heal_amount = max(1, int(agent.max_hp) // 2)
            if heal_amount > 0:
                agent.hp = min(int(agent.max_hp), int(agent.hp) + heal_amount)
            events.append(f"skill_up:{skill}:{level}")
        agent.skill_xp[skill] = current
        agent.skills[skill] = level

    def _skill_xp_to_next(self, level: int) -> int:
        return 20 + 15 * max(0, int(level))

    def _overall_level(self, agent: AgentState) -> int:
        return int(sum(max(0, int(v)) for v in agent.skills.values()))

    def _carried_weight(self, agent: AgentState) -> float:
        weight = 0.0
        for item in agent.inventory + agent.equipped:
            weight += float(self.item_weight.get(self._item_base_id(item), 1.0))
        return weight

    def _carry_capacity(self, agent: AgentState, athletics_level: int | None = None) -> float:
        ath = self._skill_level(agent, "athletics") if athletics_level is None else athletics_level
        return 8.0 + 1.3 * float(agent.strength) + 0.8 * float(ath)

    def _encumbrance_ratio(self, agent: AgentState) -> float:
        cap = max(1e-6, self._carry_capacity(agent))
        return self._carried_weight(agent) / cap

    def _encumbrance_penalty(self, agent: AgentState, athletics_level: int) -> float:
        ratio = self._encumbrance_ratio(agent)
        if ratio <= 1.0:
            return 0.0
        mitigation = min(0.8, 0.08 * athletics_level)
        return 0.02 * (ratio - 1.0) * (1.0 - mitigation)

    def _hit_chance(
        self, attacker: AgentState, target: AgentState, skill_level: int
    ) -> float:
        chance = 0.72
        chance += 0.016 * float(attacker.dexterity - 5)
        chance += 0.02 * float(skill_level - 1)
        chance -= 0.014 * float(target.dexterity - 5)
        if self._has_adjacent_ally(attacker):
            chance += float(self.config.formation_hit_bonus)
        if self._has_adjacent_ally(target):
            chance -= 0.5 * float(self.config.formation_hit_bonus)
        return max(0.2, min(0.95, chance))

    def _damage_stat_bonus(self, attacker: AgentState, damage_type: str) -> int:
        if damage_type == DAMAGE_TYPE_PIERCE:
            return max(0, (attacker.dexterity - 5) // 3)
        return max(0, (attacker.strength - 5) // 3)

    def _nearby_chest_counts(
        self, center: Tuple[int, int], height: int, width: int
    ) -> Dict[str, int]:
        assert self.state is not None
        counts = {"closed": 0, "opened": 0}
        cr, cc = center
        start_r = cr - (height // 2)
        start_c = cc - (width // 2)
        for r in range(start_r, start_r + height):
            for c in range(start_c, start_c + width):
                chest = self.state.chests.get((r, c))
                if chest is None:
                    continue
                if chest.opened:
                    counts["opened"] += 1
                else:
                    counts["closed"] += 1
        return counts

    def _nearby_resource_counts(
        self, center: Tuple[int, int], height: int, width: int
    ) -> Dict[str, int]:
        assert self.state is not None
        cr, cc = center
        start_r = cr - (height // 2)
        start_c = cc - (width // 2)
        out: Dict[str, int] = {}
        for r in range(start_r, start_r + height):
            for c in range(start_c, start_c + width):
                node = self.state.resource_nodes.get((r, c))
                if node is None or int(node.remaining) <= 0:
                    continue
                out[node.node_id] = out.get(node.node_id, 0) + 1
        return out

    def _nearby_station_counts(
        self, center: Tuple[int, int], height: int, width: int
    ) -> Dict[str, int]:
        assert self.state is not None
        cr, cc = center
        start_r = cr - (height // 2)
        start_c = cc - (width // 2)
        out: Dict[str, int] = {}
        for r in range(start_r, start_r + height):
            for c in range(start_c, start_c + width):
                station = self.state.stations.get((r, c))
                if station is None:
                    continue
                out[station.station_id] = out.get(station.station_id, 0) + 1
        return out

    def _nearby_agent_counts(
        self, aid: str, center: Tuple[int, int], height: int, width: int
    ) -> Dict[str, object]:
        assert self.state is not None
        cr, cc = center
        start_r = cr - (height // 2)
        start_c = cc - (width // 2)
        end_r = start_r + height - 1
        end_c = start_c + width - 1
        total_visible = 0
        adjacent = 0
        nearest: int | None = None
        relation_counts = {"ally": 0, "neutral": 0, "enemy": 0}
        observed_agents: List[Dict[str, object]] = []
        actor = self.state.agents[aid]
        for other_id, other in self.state.agents.items():
            if other_id == aid or not other.alive:
                continue
            r, c = other.position
            if r < start_r or r > end_r or c < start_c or c > end_c:
                continue
            total_visible += 1
            rel = self._faction_relation(actor, other)
            relation_counts[rel] = relation_counts.get(rel, 0) + 1
            dist = self._manhattan(center, other.position)
            if dist == 1:
                adjacent += 1
            if nearest is None or dist < nearest:
                nearest = dist
            observed_agents.append(
                {
                    "agent_id": other_id,
                    "distance": dist,
                    "position": other.position,
                    "relation": rel,
                    "hp": int(other.hp),
                    "faction_id": int(other.faction_id),
                    "overall_level": int(self._overall_level(other)),
                }
            )
        observed_agents.sort(key=lambda entry: (int(entry["distance"]), str(entry["agent_id"])))
        return {
            "visible": total_visible,
            "adjacent": adjacent,
            "nearest_distance": nearest,
            "relation_counts": relation_counts,
            "observed_agents": observed_agents,
        }

    def _nearby_monster_counts(
        self, center: Tuple[int, int], height: int, width: int
    ) -> Dict[str, object]:
        assert self.state is not None
        cr, cc = center
        start_r = cr - (height // 2)
        start_c = cc - (width // 2)
        end_r = start_r + height - 1
        end_c = start_c + width - 1
        total_visible = 0
        adjacent = 0
        nearest: int | None = None
        by_type: Dict[str, int] = {}
        for monster in self.state.monsters.values():
            if not monster.alive:
                continue
            r, c = monster.position
            if r < start_r or r > end_r or c < start_c or c > end_c:
                continue
            total_visible += 1
            by_type[monster.monster_id] = by_type.get(monster.monster_id, 0) + 1
            dist = self._manhattan(center, monster.position)
            if dist == 1:
                adjacent += 1
            if nearest is None or dist < nearest:
                nearest = dist
        return {
            "visible": total_visible,
            "adjacent": adjacent,
            "nearest_distance": nearest,
            "by_type": by_type,
        }

    def _nearby_animal_counts(
        self, center: Tuple[int, int], height: int, width: int
    ) -> Dict[str, object]:
        assert self.state is not None
        cr, cc = center
        start_r = cr - (height // 2)
        start_c = cc - (width // 2)
        end_r = start_r + height - 1
        end_c = start_c + width - 1
        total_visible = 0
        adjacent = 0
        nearest: int | None = None
        by_type: Dict[str, int] = {}
        for animal in self.state.animals.values():
            if not animal.alive:
                continue
            r, c = animal.position
            if r < start_r or r > end_r or c < start_c or c > end_c:
                continue
            total_visible += 1
            by_type[animal.animal_id] = by_type.get(animal.animal_id, 0) + 1
            dist = self._manhattan(center, animal.position)
            if dist == 1:
                adjacent += 1
            if nearest is None or dist < nearest:
                nearest = dist
        return {
            "visible": total_visible,
            "adjacent": adjacent,
            "nearest_distance": nearest,
            "by_type": by_type,
        }

    def _has_adjacent_hostile(self, agent: AgentState, actor_id: str) -> bool:
        assert self.state is not None
        ar, ac = agent.position
        for monster in self.state.monsters.values():
            if not monster.alive:
                continue
            if abs(monster.position[0] - ar) + abs(monster.position[1] - ac) == 1:
                return True
        for other_id, other in self.state.agents.items():
            if other_id == actor_id or not other.alive:
                continue
            if self._is_allied(agent, other):
                continue
            if abs(other.position[0] - ar) + abs(other.position[1] - ac) == 1:
                return True
        return False

    def _cluster_agent_starts_for_combat(
        self, grid: List[List[str]], starts: List[Tuple[int, int]]
    ) -> List[Tuple[int, int]]:
        if len(starts) <= 1:
            return starts
        occupied = {starts[0]}
        out = [starts[0]]
        base = starts[0]
        ring = [
            (base[0] - 1, base[1]),
            (base[0] + 1, base[1]),
            (base[0], base[1] - 1),
            (base[0], base[1] + 1),
            (base[0] - 1, base[1] - 1),
            (base[0] - 1, base[1] + 1),
            (base[0] + 1, base[1] - 1),
            (base[0] + 1, base[1] + 1),
        ]
        fallback = list(starts[1:])
        for _ in starts[1:]:
            placed = None
            for r, c in ring:
                if r < 0 or c < 0 or r >= len(grid) or c >= len(grid[0]):
                    continue
                if (r, c) in occupied:
                    continue
                tile_id = grid[r][c]
                if not self.tiles[tile_id].walkable:
                    continue
                placed = (r, c)
                break
            if placed is None:
                while fallback:
                    cand = fallback.pop(0)
                    if cand not in occupied:
                        placed = cand
                        break
            if placed is None:
                placed = starts[1]
            out.append(placed)
            occupied.add(placed)
        return out

    def _apply_monster_turn(
        self,
        rewards: Dict[str, float],
        terminations: Dict[str, bool],
        info: Dict[str, Dict[str, object]],
    ) -> None:
        assert self.state is not None
        if not self.state.monsters:
            return

        for entity_id in sorted(self.state.monsters.keys()):
            monster = self.state.monsters[entity_id]
            if not monster.alive or monster.hp <= 0:
                monster.alive = False
                continue

            target_id = self._nearest_alive_agent_id(
                monster.position,
                max_range=max(1, int(self.config.monster_sight_range)),
            )
            if target_id is None:
                self._monster_random_move(monster, info)
                continue
            target = self.state.agents[target_id]
            dist = self._manhattan(monster.position, target.position)
            if dist == 1:
                self._monster_attack(monster, target, rewards, terminations, info)
            else:
                self._monster_move_toward_target(monster, target.position, info)

    def _tick_fires(self) -> None:
        assert self.state is not None
        floor_id = (
            self.mapgen_cfg.floor_fallback_id
            if self.mapgen_cfg.floor_fallback_id in self.tiles
            else "floor"
        )
        to_extinguish: List[Tuple[int, int]] = []
        for r, row in enumerate(self.state.grid):
            for c, tile_id in enumerate(row):
                if tile_id != "campfire":
                    continue
                fuel = int(self.state.tile_interactions.get((r, c), 0))
                fuel = max(0, fuel - FIRE_FUEL_DECAY_PER_STEP)
                if fuel <= 0:
                    to_extinguish.append((r, c))
                    continue
                self.state.tile_interactions[(r, c)] = fuel
        for pos in to_extinguish:
            self.state.grid[pos[0]][pos[1]] = floor_id
            self.state.tile_interactions.pop(pos, None)

    def _apply_animal_turn(self, info: Dict[str, Dict[str, object]]) -> None:
        assert self.state is not None
        if not self.state.animals:
            return
        for entity_id in sorted(self.state.animals.keys()):
            animal = self.state.animals[entity_id]
            if not animal.alive:
                continue
            animal.age = int(animal.age) + 1
            if animal.reproduction_cooldown > 0:
                animal.reproduction_cooldown = int(animal.reproduction_cooldown) - 1
            if animal.sheared and animal.wool_regrow > 0:
                animal.wool_regrow = int(animal.wool_regrow) - 1
                if animal.wool_regrow <= 0:
                    animal.sheared = False
            steps = max(1, int(animal.movement_speed))
            for _ in range(steps):
                if not animal.alive:
                    break
                if not self._animal_hunt_step(animal):
                    self._animal_move(animal)
            self._animal_tick_needs(animal)
            if not animal.alive:
                continue
            self._animal_try_reproduce(animal)

    def _animal_tick_needs(self, animal: AnimalState) -> None:
        assert self.state is not None
        animal.hunger = max(0, int(animal.hunger) - 1)
        animal.thirst = max(0, int(animal.thirst) - 1)
        if self._animal_consume_food(animal):
            animal.hunger = min(int(animal.max_hunger), int(animal.hunger) + 2)
        if self._animal_can_drink(animal.position):
            animal.thirst = min(int(animal.max_thirst), int(animal.thirst) + 2)
        if animal.hunger <= 0 or animal.thirst <= 0:
            animal.alive = False
            self._drop_animal_material(animal, [])

    def _animal_can_eat(self, animal: AnimalState, pos: Tuple[int, int]) -> bool:
        assert self.state is not None
        if bool(animal.carnivore):
            return self._animal_find_prey(animal) is not None
        forage_tiles = {"grass", "bush", "tree"}
        for nr, nc in self._animal_neighborhood(pos):
            if nr < 0 or nc < 0 or nr >= len(self.state.grid) or nc >= len(self.state.grid[0]):
                continue
            tile_id = self.state.grid[nr][nc]
            if tile_id not in forage_tiles:
                continue
            max_interactions = max(1, int(self.tiles[tile_id].max_interactions))
            used = int(self.state.tile_interactions.get((nr, nc), 0))
            if used < max_interactions:
                return True
        return False

    def _animal_consume_food(self, animal: AnimalState) -> bool:
        if bool(animal.carnivore):
            return False
        return self._animal_consume_forage(animal.position)

    def _animal_consume_forage(self, pos: Tuple[int, int]) -> bool:
        assert self.state is not None
        forage_tiles = {"grass", "bush", "tree"}
        candidates = self._animal_neighborhood(pos)
        self._rng.shuffle(candidates)
        for nr, nc in candidates:
            if nr < 0 or nc < 0 or nr >= len(self.state.grid) or nc >= len(self.state.grid[0]):
                continue
            tile_id = self.state.grid[nr][nc]
            if tile_id not in forage_tiles:
                continue
            max_interactions = max(1, int(self.tiles[tile_id].max_interactions))
            used = int(self.state.tile_interactions.get((nr, nc), 0))
            if used >= max_interactions:
                continue
            used += 1
            if used >= max_interactions:
                floor_id = (
                    self.mapgen_cfg.floor_fallback_id
                    if self.mapgen_cfg.floor_fallback_id in self.tiles
                    else "floor"
                )
                self.state.grid[nr][nc] = floor_id
                self.state.tile_interactions.pop((nr, nc), None)
            else:
                self.state.tile_interactions[(nr, nc)] = used
            return True
        return False

    def _animal_neighborhood(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        r, c = pos
        return [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]

    def _animal_can_drink(self, pos: Tuple[int, int]) -> bool:
        assert self.state is not None
        r, c = pos
        if self.state.grid[r][c] in WATER_TILE_IDS:
            return True
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nr < 0 or nc < 0 or nr >= len(self.state.grid) or nc >= len(self.state.grid[0]):
                continue
            if self.state.grid[nr][nc] in WATER_TILE_IDS:
                return True
        return False

    def _animal_move(self, animal: AnimalState) -> None:
        assert self.state is not None
        r, c = animal.position
        candidates = [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
        walkable = [(nr, nc) for (nr, nc) in candidates if self._walkable_for_animal(nr, nc, animal.entity_id)]
        if not walkable:
            return
        hungry = int(animal.hunger) <= max(2, int(animal.max_hunger) // 3)
        thirsty = int(animal.thirst) <= max(2, int(animal.max_thirst) // 3)
        if thirsty:
            near_water = [p for p in walkable if self._animal_can_drink(p)]
            if near_water:
                animal.position = self._rng.choice(near_water)
                return
        if hungry:
            near_food = [p for p in walkable if self._animal_can_eat(animal, p)]
            if near_food:
                animal.position = self._rng.choice(near_food)
                return
        animal.position = self._rng.choice(walkable)

    def _animal_find_prey(self, predator: AnimalState) -> AnimalState | None:
        assert self.state is not None
        if not bool(predator.carnivore):
            return None
        best: AnimalState | None = None
        best_dist: int | None = None
        for other in self.state.animals.values():
            if not other.alive or other.entity_id == predator.entity_id:
                continue
            if other.animal_id == predator.animal_id:
                continue
            if int(predator.prey_score) < int(other.prey_score) + PREY_SCORE_HUNT_MARGIN:
                continue
            dist = self._manhattan(predator.position, other.position)
            if dist > max(4, int(self.config.monster_sight_range)):
                continue
            if not self._has_line_of_sight(predator.position, other.position):
                continue
            if best_dist is None or dist < best_dist:
                best = other
                best_dist = dist
        return best

    def _animal_hunt_step(self, predator: AnimalState) -> bool:
        assert self.state is not None
        target = self._animal_find_prey(predator)
        if target is None:
            return False
        dist = self._manhattan(predator.position, target.position)
        if dist <= 1:
            self._animal_attack_animal(predator, target)
            return True
        pr, pc = predator.position
        tr, tc = target.position
        candidates = [(pr - 1, pc), (pr + 1, pc), (pr, pc - 1), (pr, pc + 1)]
        walkable = [
            (nr, nc)
            for nr, nc in candidates
            if self._walkable_for_animal(nr, nc, predator.entity_id)
        ]
        if not walkable:
            return True
        walkable.sort(key=lambda p: self._manhattan(p, (tr, tc)))
        predator.position = walkable[0]
        if self._manhattan(predator.position, target.position) <= 1:
            self._animal_attack_animal(predator, target)
        return True

    def _animal_attack_animal(self, predator: AnimalState, prey: AnimalState) -> None:
        if not predator.alive or not prey.alive:
            return
        damage = max(1, int(predator.hp // 2) + int(predator.prey_score // 2))
        prey.hp = max(0, int(prey.hp) - damage)
        if prey.hp > 0:
            return
        prey.alive = False
        predator.hunger = min(int(predator.max_hunger), int(predator.hunger) + max(3, int(prey.max_hunger // 3)))

    def _animal_try_reproduce(self, animal: AnimalState) -> None:
        assert self.state is not None
        if not animal.alive:
            return
        if animal.age < animal.mature_age:
            return
        if animal.reproduction_cooldown > 0:
            return
        if animal.hunger < max(3, animal.max_hunger // 2):
            return
        if animal.thirst < max(3, animal.max_thirst // 2):
            return
        partner = None
        for other in self.state.animals.values():
            if other.entity_id == animal.entity_id or not other.alive:
                continue
            if other.animal_id != animal.animal_id:
                continue
            if str(other.gender) == str(animal.gender):
                continue
            if other.age < other.mature_age or other.reproduction_cooldown > 0:
                continue
            if self._manhattan(animal.position, other.position) <= 1:
                partner = other
                break
        if partner is None:
            return
        litter_min = max(1, int(min(animal.litter_size_min, animal.litter_size_max)))
        litter_max = max(1, int(max(animal.litter_size_min, animal.litter_size_max)))
        litter = self._rng.randint(litter_min, litter_max)
        parent_def = self.animals.get(animal.animal_id)
        if parent_def is None:
            return
        ar, ac = animal.position
        offsets = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1)]
        self._rng.shuffle(offsets)
        spawned = 0
        for dr, dc in offsets:
            if spawned >= litter:
                break
            nr, nc = ar + dr, ac + dc
            if not self._walkable_for_animal(nr, nc, moving_entity_id=""):
                continue
            entity_id = f"animal_{len(self.state.animals)}"
            self.state.animals[entity_id] = AnimalState(
                entity_id=entity_id,
                animal_id=parent_def.animal_id,
                name=parent_def.name,
                symbol=parent_def.symbol,
                color=parent_def.color,
                position=(nr, nc),
                hp=parent_def.hp,
                max_hp=parent_def.hp,
                hunger=max(2, int(parent_def.max_hunger // 2)),
                max_hunger=parent_def.max_hunger,
                thirst=max(2, int(parent_def.max_thirst // 2)),
                max_thirst=parent_def.max_thirst,
                age=0,
                mature_age=parent_def.mature_age,
                reproduction_cooldown=parent_def.reproduction_cooldown,
                reproduction_cooldown_max=parent_def.reproduction_cooldown,
                prey_score=parent_def.prey_score,
                movement_speed=parent_def.movement_speed,
                carnivore=parent_def.carnivore,
                gender=str(self._rng.choice(["female", "male"])),
                litter_size_min=parent_def.litter_size_min,
                litter_size_max=parent_def.litter_size_max,
                can_shear=parent_def.can_shear,
                sheared=False,
                shear_item=parent_def.shear_item,
                wool_regrow=0,
                shear_regrow_max=parent_def.shear_regrow_steps,
                alive=True,
            )
            spawned += 1
        if spawned <= 0:
            return
        animal.reproduction_cooldown = animal.reproduction_cooldown_max
        partner.reproduction_cooldown = partner.reproduction_cooldown_max

    def _walkable_for_animal(
        self, r: int, c: int, moving_entity_id: str
    ) -> bool:
        assert self.state is not None
        if not self._walkable(r, c):
            return False
        for animal in self.state.animals.values():
            if (
                animal.alive
                and animal.entity_id != moving_entity_id
                and animal.position == (r, c)
            ):
                return False
        return True

    def _nearest_alive_agent_id(
        self, position: Tuple[int, int], max_range: int | None = None
    ) -> str | None:
        assert self.state is not None
        best_id: str | None = None
        best_dist: int | None = None
        for aid in self.possible_agents:
            agent = self.state.agents[aid]
            if not agent.alive or agent.hp <= 0:
                continue
            dist = self._manhattan(position, agent.position)
            if max_range is not None and dist > int(max_range):
                continue
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_id = aid
        return best_id

    def _monster_attack(
        self,
        monster: MonsterState,
        target: AgentState,
        rewards: Dict[str, float],
        terminations: Dict[str, bool],
        info: Dict[str, Dict[str, object]],
    ) -> None:
        hit_chance = 0.62 + 0.03 * float(monster.acc)
        hit_chance -= 0.02 * float(target.dexterity - 5)
        hit_chance -= 0.01 * float(self._skill_level(target, "athletics"))
        hit_chance = max(0.1, min(0.95, hit_chance))

        target_events = info[target.agent_id]["events"]
        target_events.append(
            f"monster_attack:{monster.monster_id}:{monster.entity_id}:roll"
        )
        if self._rng.random() > hit_chance:
            target_events.append(f"monster_miss:{monster.monster_id}:{monster.entity_id}")
            return

        raw_damage = self._rng.randint(monster.dmg_min, max(monster.dmg_min, monster.dmg_max))
        dr, hit_slot, armor_mitigation, armor_skill = self._roll_hit_location_dr(
            target, DAMAGE_TYPE_BLUNT
        )
        final_damage = max(0, raw_damage - dr)
        final_damage = self._apply_guard_reduction(target, final_damage, target_events)
        target_events.append(
            f"monster_hit_roll:{monster.monster_id}:raw:{raw_damage}:slot:{hit_slot}:dr:{dr}:final:{final_damage}"
        )
        if armor_skill and armor_mitigation > 0 and final_damage < raw_damage:
            self._gain_skill_xp(
                target,
                armor_skill,
                max(1, armor_mitigation // 2),
                target_events,
            )
        if final_damage <= 0:
            target_events.append(f"monster_blocked:{monster.monster_id}")
            return

        target.hp = max(0, target.hp - final_damage)
        rewards[target.agent_id] -= 0.03 * final_damage
        target_events.append(f"monster_hit:{monster.monster_id}:{final_damage}")
        if target.hp > 0 and self._rng.random() < 0.14:
            self._apply_status(
                target_aid=target.agent_id,
                status_id="poison",
                source_aid=monster.entity_id,
                info=info,
                rewards=rewards,
            )
        if target.hp <= 0 and target.alive:
            target.alive = False
            terminations[target.agent_id] = True
            rewards[target.agent_id] -= 1.0
            target_events.append("death")
            target_events.append(f"death_by_monster:{monster.monster_id}")

    def _monster_move_toward_target(
        self,
        monster: MonsterState,
        target_pos: Tuple[int, int],
        info: Dict[str, Dict[str, object]],
    ) -> None:
        assert self.state is not None
        current_dist = self._manhattan(monster.position, target_pos)
        if current_dist <= 1:
            return

        mr, mc = monster.position
        candidates = [
            (mr - 1, mc),
            (mr + 1, mc),
            (mr, mc - 1),
            (mr, mc + 1),
        ]
        best_moves: List[Tuple[int, int]] = []
        best_dist = current_dist
        for nr, nc in candidates:
            if not self._walkable_for_monster(nr, nc, monster.entity_id):
                continue
            d = self._manhattan((nr, nc), target_pos)
            if d < best_dist:
                best_dist = d
                best_moves = [(nr, nc)]
            elif d == best_dist:
                best_moves.append((nr, nc))
        if best_moves:
            old = monster.position
            monster.position = self._rng.choice(best_moves)
            nearest = self._nearest_alive_agent_id(monster.position)
            if nearest is not None:
                info[nearest]["events"].append(
                    f"monster_move:{monster.monster_id}:{old}->{monster.position}"
                )

    def _monster_random_move(
        self,
        monster: MonsterState,
        info: Dict[str, Dict[str, object]],
    ) -> None:
        assert self.state is not None
        mr, mc = monster.position
        candidates = [
            (mr - 1, mc),
            (mr + 1, mc),
            (mr, mc - 1),
            (mr, mc + 1),
        ]
        walkable = [
            (r, c)
            for (r, c) in candidates
            if self._walkable_for_monster(r, c, monster.entity_id)
        ]
        if not walkable:
            return
        old = monster.position
        monster.position = self._rng.choice(walkable)
        nearest = self._nearest_alive_agent_id(monster.position)
        if nearest is not None:
            info[nearest]["events"].append(
                f"monster_wander:{monster.monster_id}:{old}->{monster.position}"
            )

    def _drop_monster_loot(self, monster: MonsterState, events: List[str]) -> None:
        assert self.state is not None
        mdef = self.monsters.get(monster.monster_id)
        if mdef is None or not mdef.loot:
            return
        weights = [max(0.0, float(entry.weight)) for entry in mdef.loot]
        if sum(weights) <= 0:
            return
        picked = self._rng.choices(mdef.loot, weights=weights, k=1)[0]
        qty_min = min(picked.min_qty, picked.max_qty)
        qty_max = max(picked.min_qty, picked.max_qty)
        qty = self._rng.randint(qty_min, qty_max)
        if qty <= 0:
            return
        pos = monster.position
        bag = self.state.ground_items.setdefault(pos, [])
        for _ in range(qty):
            bag.append(picked.item)
        events.append(f"monster_loot_drop:{monster.monster_id}:{picked.item}:{qty}")

    def _drop_animal_material(self, animal: AnimalState, events: List[str]) -> None:
        assert self.state is not None
        adef = self.animals.get(animal.animal_id)
        if adef is None or not adef.drop_item:
            return
        qty = 1 if animal.animal_id != "chicken" else 2
        bag = self.state.ground_items.setdefault(animal.position, [])
        for _ in range(max(1, qty)):
            bag.append(adef.drop_item)
        if events is not None:
            events.append(f"animal_drop:{animal.animal_id}:{adef.drop_item}:{qty}")

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


class PettingZooParallelRLRLGym(MultiAgentRLRLGym):
    """Primary env class name to signal PettingZoo Parallel-style usage."""
