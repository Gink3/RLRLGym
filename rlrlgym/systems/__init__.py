"""Game/system helpers namespace."""

from ..curriculum import load_curriculum_phases
from ..featurize import observation_vector_size, vectorize_observation
from ..names import generate_full_name, load_name_table
from ..scenario import (
    Scenario,
    ScenarioAgent,
    agent_combined_payload,
    apply_scenario_to_env_config,
    estimate_max_networks,
    load_scenario,
    make_all_race_class_combinations,
    save_scenario,
)

__all__ = [
    "Scenario",
    "ScenarioAgent",
    "load_curriculum_phases",
    "observation_vector_size",
    "vectorize_observation",
    "generate_full_name",
    "load_name_table",
    "agent_combined_payload",
    "apply_scenario_to_env_config",
    "estimate_max_networks",
    "load_scenario",
    "make_all_race_class_combinations",
    "save_scenario",
]
