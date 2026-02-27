"""RLRLGym package."""

from .classes import AgentClass, load_classes
from .env import EnvConfig, MultiAgentRLRLGym
from .env import PettingZooParallelRLRLGym
from .featurize import observation_vector_size, vectorize_observation
from .items import ItemCatalog, ItemDef, WeaponDef, load_items
from .mapgen_config import MapGenConfig, load_mapgen_config
from .monsters import load_monster_spawns, load_monsters
from .profiles import AgentProfile, load_profiles
from .races import AgentRace, load_races
from .render import PlaybackController, RenderWindow
from .scenario import (
    Scenario,
    ScenarioAgent,
    agent_combined_payload,
    apply_scenario_to_env_config,
    estimate_max_networks,
    load_scenario,
    make_all_race_class_combinations,
    save_scenario,
)
from .vector_env import SyncVectorEnv

__all__ = [
    "EnvConfig",
    "MultiAgentRLRLGym",
    "PettingZooParallelRLRLGym",
    "vectorize_observation",
    "observation_vector_size",
    "AgentProfile",
    "load_profiles",
    "AgentClass",
    "load_classes",
    "AgentRace",
    "load_races",
    "ItemCatalog",
    "ItemDef",
    "WeaponDef",
    "load_items",
    "load_monsters",
    "load_monster_spawns",
    "MapGenConfig",
    "load_mapgen_config",
    "PlaybackController",
    "RenderWindow",
    "Scenario",
    "ScenarioAgent",
    "load_scenario",
    "save_scenario",
    "make_all_race_class_combinations",
    "apply_scenario_to_env_config",
    "agent_combined_payload",
    "estimate_max_networks",
    "SyncVectorEnv",
]
