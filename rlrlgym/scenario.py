"""Scenario definitions and helpers for composing race+class agent rosters."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SCENARIO_ENV_FILE = "env_config.json"
SCENARIO_AGENTS_FILE = "agents.json"


@dataclass
class ScenarioAgent:
    """Single agent entry in a scenario."""

    agent_id: str
    race: str
    class_name: str
    name: Optional[str] = None
    profile: Optional[str] = None
    network: Optional[str] = None
    observation_config: Dict[str, object] = field(default_factory=dict)


@dataclass
class Scenario:
    """Scenario file payload."""

    name: str
    env_config: Dict[str, object] = field(default_factory=dict)
    agents: List[ScenarioAgent] = field(default_factory=list)


def _normalize_agent_id(index: int) -> str:
    return f"agent_{index}"


def _agent_from_row(row: Dict[str, object], index: int) -> ScenarioAgent:
    if not isinstance(row, dict):
        raise ValueError(f"scenario.agents[{index}] must be an object")
    race = str(row.get("race", "")).strip()
    class_name = str(row.get("class", row.get("class_name", ""))).strip()
    if not race:
        raise ValueError(f"scenario.agents[{index}] missing 'race'")
    if not class_name:
        raise ValueError(f"scenario.agents[{index}] missing 'class'")
    name_raw = row.get("name")
    profile_raw = row.get("profile")
    network_raw = row.get("network")
    obs_raw = row.get("observation_config", {})
    if obs_raw is None:
        obs_raw = {}
    if not isinstance(obs_raw, dict):
        raise ValueError(f"scenario.agents[{index}].observation_config must be an object")
    return ScenarioAgent(
        agent_id=_normalize_agent_id(index),
        race=race,
        class_name=class_name,
        name=(str(name_raw).strip() if name_raw not in (None, "") else None),
        profile=(str(profile_raw).strip() if profile_raw not in (None, "") else None),
        network=(str(network_raw).strip() if network_raw not in (None, "") else None),
        observation_config=dict(obs_raw),
    )


def _load_split_scenario_dir(path: Path) -> Scenario:
    env_path = path / SCENARIO_ENV_FILE
    agents_path = path / SCENARIO_AGENTS_FILE
    if not env_path.exists():
        raise ValueError(f"Missing scenario env config file: {env_path}")
    if not agents_path.exists():
        raise ValueError(f"Missing scenario agents file: {agents_path}")

    env_raw = json.loads(env_path.read_text(encoding="utf-8"))
    if not isinstance(env_raw, dict):
        raise ValueError("Scenario env_config JSON must be an object")
    if "schema_version" not in env_raw or not isinstance(env_raw["schema_version"], int):
        raise ValueError("Scenario env_config JSON requires integer schema_version")
    env_cfg = env_raw.get("env_config", env_raw)
    if not isinstance(env_cfg, dict):
        raise ValueError("Scenario env_config payload must be an object")

    agents_raw = json.loads(agents_path.read_text(encoding="utf-8"))
    if not isinstance(agents_raw, dict):
        raise ValueError("Scenario agents JSON must be an object")
    if "schema_version" not in agents_raw or not isinstance(agents_raw["schema_version"], int):
        raise ValueError("Scenario agents JSON requires integer schema_version")
    rows = agents_raw.get("agents", [])
    if not isinstance(rows, list):
        raise ValueError("Scenario agents JSON requires array 'agents'")
    if not rows:
        raise ValueError("Scenario agents must include at least one agent")

    agents = [_agent_from_row(row, idx) for idx, row in enumerate(rows)]
    return Scenario(name=str(path.name), env_config=dict(env_cfg), agents=agents)


def _load_legacy_single_file(path: Path) -> Scenario:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Scenario JSON must be an object")
    if "schema_version" not in payload or not isinstance(payload["schema_version"], int):
        raise ValueError("Scenario JSON requires integer schema_version")
    body = payload.get("scenario", payload)
    if not isinstance(body, dict):
        raise ValueError("Scenario payload must be an object")
    name = str(body.get("name", path.stem)).strip() or path.stem
    env_config = body.get("env_config", {})
    if env_config is None:
        env_config = {}
    if not isinstance(env_config, dict):
        raise ValueError("scenario.env_config must be an object")
    raw_agents = body.get("agents", [])
    if not isinstance(raw_agents, list):
        raise ValueError("scenario.agents must be an array")
    if not raw_agents:
        raise ValueError("scenario.agents must include at least one agent")
    agents = [_agent_from_row(row, idx) for idx, row in enumerate(raw_agents)]
    return Scenario(name=name, env_config=dict(env_config), agents=agents)


def load_scenario(path: str | Path) -> Scenario:
    p = Path(path)
    if p.is_dir():
        return _load_split_scenario_dir(p)
    if p.is_file():
        split_dir = p.with_suffix("")
        if split_dir.exists() and split_dir.is_dir():
            return _load_split_scenario_dir(split_dir)
        return _load_legacy_single_file(p)
    raise ValueError(f"Scenario path not found: {p}")


def save_scenario(path: str | Path, scenario: Scenario) -> Path:
    p = Path(path)
    scenario_dir = p.with_suffix("") if p.suffix else p
    scenario_dir.mkdir(parents=True, exist_ok=True)

    env_payload = {
        "schema_version": 1,
        "env_config": dict(scenario.env_config),
    }
    agents_payload = {
        "schema_version": 1,
        "name": scenario.name,
        "agents": [
            {
                "agent_id": agent.agent_id,
                "race": agent.race,
                "class": agent.class_name,
                "name": agent.name,
                "profile": agent.profile,
                "network": agent.network,
                "observation_config": dict(agent.observation_config),
            }
            for agent in scenario.agents
        ],
    }
    (scenario_dir / SCENARIO_ENV_FILE).write_text(
        json.dumps(env_payload, indent=2), encoding="utf-8"
    )
    (scenario_dir / SCENARIO_AGENTS_FILE).write_text(
        json.dumps(agents_payload, indent=2), encoding="utf-8"
    )
    return scenario_dir


def make_all_race_class_combinations(
    races: List[str],
    classes: List[str],
    *,
    default_profile_by_race: Optional[Dict[str, str]] = None,
    default_network: Optional[str] = None,
) -> List[ScenarioAgent]:
    out: List[ScenarioAgent] = []
    for race in sorted(str(r).strip() for r in races if str(r).strip()):
        for cls in sorted(str(c).strip() for c in classes if str(c).strip()):
            idx = len(out)
            out.append(
                ScenarioAgent(
                    agent_id=_normalize_agent_id(idx),
                    race=race,
                    class_name=cls,
                    name=None,
                    profile=(default_profile_by_race or {}).get(race),
                    network=default_network,
                    observation_config={},
                )
            )
    return out


def agent_combined_payload(
    *,
    agent_id: str,
    race: str,
    class_name: str,
    name: Optional[str],
    profile: Optional[str],
    network: Optional[str],
    observation_config: Optional[Dict[str, object]],
    race_row: Optional[Dict[str, object]] = None,
    class_row: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    return {
        "agent_id": str(agent_id),
        "race": str(race),
        "class": str(class_name),
        "name": name,
        "profile": profile,
        "network": network,
        "observation_config": dict(observation_config or {}),
        "race_def": dict(race_row or {}),
        "class_def": dict(class_row or {}),
    }


def apply_scenario_to_env_config(env_config, scenario: Scenario):
    """Return a copied EnvConfig with scenario overrides/maps applied."""

    cfg = copy.deepcopy(env_config)
    for key, value in scenario.env_config.items():
        if hasattr(cfg, key):
            setattr(cfg, key, copy.deepcopy(value))

    cfg.n_agents = len(scenario.agents)
    cfg.agent_profile_map = {}
    cfg.agent_race_map = {}
    cfg.agent_class_map = {}
    cfg.agent_observation_config = {}

    normalized_agents: List[Dict[str, object]] = []
    for idx, agent in enumerate(scenario.agents):
        aid = _normalize_agent_id(idx)
        cfg.agent_race_map[aid] = agent.race
        cfg.agent_class_map[aid] = agent.class_name
        if agent.profile:
            cfg.agent_profile_map[aid] = agent.profile
        if agent.observation_config:
            cfg.agent_observation_config[aid] = dict(agent.observation_config)
        normalized_agents.append(
            {
                "agent_id": aid,
                "race": agent.race,
                "class": agent.class_name,
                "name": agent.name,
                "profile": agent.profile,
                "network": agent.network,
                "observation_config": dict(agent.observation_config),
            }
        )
    cfg.agent_scenario = normalized_agents
    return cfg


def estimate_max_networks(
    *,
    per_network_params: int,
    available_ram_bytes: int,
    usable_fraction: float = 0.45,
    bytes_per_param: int = 32,
) -> Tuple[int, int]:
    """Estimate max concurrently resident Python MLP policies.

    Returns `(max_networks, estimated_bytes_per_network)`.
    """

    params = max(1, int(per_network_params))
    avail = max(0, int(available_ram_bytes))
    bpp = max(8, int(bytes_per_param))
    usable = int(float(avail) * float(max(0.05, min(0.9, usable_fraction))))
    per = params * bpp
    return max(1, usable // max(1, per)), per
