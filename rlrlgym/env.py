"""PettingZoo-style parallel multi-agent roguelike environment."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .constants import (
    ACTION_ATTACK,
    ACTION_EAT,
    ACTION_EQUIP,
    ACTION_INTERACT,
    ACTION_LOOT,
    ACTION_MOVE_EAST,
    ACTION_MOVE_NORTH,
    ACTION_MOVE_SOUTH,
    ACTION_MOVE_WEST,
    ACTION_PICKUP,
    ACTION_USE,
    ACTION_WAIT,
    MOVE_DELTAS,
)
from .classes import AgentClass, load_classes
from .items import ItemCatalog, load_items
from .mapgen import generate_map, sample_walkable_positions
from .mapgen_config import MapGenConfig, load_mapgen_config
from .models import AgentState, ChestState, EnvState, MonsterState
from .monsters import MonsterDef, MonsterSpawnEntry, load_monster_spawns, load_monsters
from .profiles import AgentProfile, load_profiles
from .races import AgentRace, load_races
from .render import RenderWindow
from .tiles import load_tileset

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
HIT_SLOT_TO_ARMOR_SKILL: Dict[str, str] = {
    "head": "armor_head",
    "chest": "armor_chest",
    "back": "armor_back",
    "arms": "armor_arms",
    "legs": "armor_legs",
    "neck": "armor_neck",
    "rings": "armor_rings",
}

@dataclass
class EnvConfig:
    width: int = 50
    height: int = 50
    max_steps: int = 150
    n_agents: int = 2
    tiles_path: str = str(Path("data") / "tiles.json")
    profiles_path: str = str(Path("data") / "agent_profiles.json")
    races_path: str = str(Path("data") / "agent_races.json")
    classes_path: str = str(Path("data") / "agent_classes.json")
    items_path: str = str(Path("data") / "items.json")
    monsters_path: str = str(Path("data") / "monsters.json")
    monster_spawns_path: str = str(Path("data") / "monster_spawns.json")
    mapgen_config_path: str = str(Path("data") / "mapgen_config.json")
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
    render_enabled: bool = True

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
        merged["agent_profile_map"] = dict(merged.get("agent_profile_map", {}))
        merged["agent_race_map"] = dict(merged.get("agent_race_map", {}))
        merged["agent_class_map"] = dict(merged.get("agent_class_map", {}))
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
        self.tiles = load_tileset(self.config.tiles_path)
        self.profiles: Dict[str, AgentProfile] = load_profiles(
            self.config.profiles_path
        )
        self.races: Dict[str, AgentRace] = load_races(self.config.races_path)
        self.classes: Dict[str, AgentClass] = load_classes(self.config.classes_path)
        self.items: ItemCatalog = load_items(self.config.items_path)
        self.monsters: Dict[str, MonsterDef] = load_monsters(self.config.monsters_path)
        self.monster_spawns: List[MonsterSpawnEntry] = load_monster_spawns(
            self.config.monster_spawns_path, self.monsters
        )
        self.weapon_damage_type = dict(self.items.weapon_damage_type)
        self.weapon_damage_range = dict(self.items.weapon_damage_range)
        self.weapon_skill_by_item = dict(self.items.weapon_skill)
        self.item_dr_bonus_vs = dict(self.items.item_dr_bonus_vs)
        self.armor_slot_by_item = dict(self.items.armor_slot_by_item)
        self.item_weight = dict(self.items.item_weight)
        self.edible_items = set(self.items.edible_items)
        self.chest_loot_table = list(self.items.chest_loot_table)
        self._validate_item_references()
        self.mapgen_cfg: MapGenConfig = load_mapgen_config(
            self.config.mapgen_config_path
        )
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

    def action_space(self, agent_id: str) -> Tuple[int, int]:
        if agent_id not in self.possible_agents:
            raise KeyError(f"Unknown agent: {agent_id}")
        return (0, 11)

    def observation_space(self, agent_id: str) -> Dict[str, object]:
        if agent_id not in self.possible_agents:
            raise KeyError(f"Unknown agent: {agent_id}")
        profile_name = self._resolve_profile_name(
            agent_id, self.possible_agents.index(agent_id)
        )
        profile = self._profile_by_name(profile_name)
        keys = ["step", "alive", "profile", "race", "class"]
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
                hunger=profile.max_hunger,
                max_hunger=profile.max_hunger,
                strength=race.strength,
                dexterity=race.dexterity,
                intellect=race.intellect,
            )
            self._apply_class_modifiers(agent, cls)
            if cls.starting_items:
                agent.inventory.extend(list(cls.starting_items))
            agent.visited.add(pos)
            agents[agent_id] = agent

        self.state = EnvState(
            grid=grid,
            tile_interactions={},
            ground_items={},
            agents=agents,
            chests={},
            step_count=0,
        )
        self.state.chests = self._spawn_chests(starts)
        self.state.monsters = self._spawn_monsters(
            occupied=starts + list(self.state.chests.keys())
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
                "action_mask": [1] * 12,
                "alive": True,
                "profile": self.state.agents[aid].profile_name,
                "race": self.state.agents[aid].race_name,
                "class": self.state.agents[aid].class_name,
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
            info[aid]["action"] = action
            delta_reward, events = self._apply_action(aid, action)
            rewards[aid] += delta_reward
            reward_components[aid]["action_total"] += float(delta_reward)
            info[aid]["events"].extend(events)
            survival_delta = self._apply_survival_costs(agent, rewards, aid, info)
            reward_components[aid]["survival"] += float(survival_delta)
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

        self._apply_monster_turn(rewards, terminations, info)
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
            dr, dc = MOVE_DELTAS[action]
            nr, nc = agent.position[0] + dr, agent.position[1] + dc
            if self._walkable(nr, nc):
                old_pos = agent.position
                agent.position = (nr, nc)
                events.append(f"move:{old_pos}->{agent.position}")
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
                armor_slot = self.armor_slot_by_item.get(item)
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
            medic_level = self._skill_level(agent, "medic")
            if "bandage" in agent.inventory and agent.hp < agent.max_hp:
                agent.inventory.remove("bandage")
                heal = 3 + (medic_level // 2) + max(0, (agent.intellect - 5) // 4)
                agent.hp = min(agent.max_hp, agent.hp + heal)
                reward += 0.12 + 0.01 * medic_level
                events.append("use:bandage")
                self._gain_skill_xp(agent, "medic", 2, events)
            elif "healing_potion" in agent.inventory and agent.hp < agent.max_hp:
                agent.inventory.remove("healing_potion")
                heal = 5 + medic_level + max(0, (agent.intellect - 5) // 3)
                agent.hp = min(agent.max_hp, agent.hp + heal)
                reward += 0.2 + 0.015 * medic_level
                events.append("use:healing_potion")
                self._gain_skill_xp(agent, "medic", 3, events)
            else:
                reward -= 0.01

        elif action == ACTION_INTERACT:
            reward += self._interact(agent, aid, events)

        elif action == ACTION_ATTACK:
            if self.config.combat_training_mode and adjacent_hostile:
                reward += 0.02
                events.append("combat_engage_bonus")
            reward += self._attack(agent, aid, events)

        return reward, events

    def _interact(self, actor: AgentState, actor_id: str, events: List[str]) -> float:
        assert self.state is not None
        reward = 0.0

        r, c = actor.position
        chest = self.state.chests.get((r, c))
        if chest and not chest.opened:
            events.append("interact:chest")
            reward += self._pickup_from_tile(actor, events)
            return reward

        tile_id = self.state.grid[r][c]
        tile = self.tiles[tile_id]

        n_interactions = self.state.tile_interactions.get((r, c), 0)
        if tile.max_interactions > 0 and n_interactions < tile.max_interactions:
            self.state.tile_interactions[(r, c)] = n_interactions + 1
            if tile_id == "shrine":
                actor.hp = min(actor.max_hp, actor.hp + 1)
                reward += 0.1
                events.append("interact:shrine")
            elif tile_id == "water":
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

    def _attack_agent(
        self, attacker: AgentState, target: AgentState, events: List[str]
    ) -> float:
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
            return -0.01

        raw_damage = self._rng.randint(damage_range[0], damage_range[1])
        raw_damage += self._damage_stat_bonus(attacker, damage_type)
        raw_damage += max(0, (skill_level - 1) // 2)
        dr, hit_slot, armor_mitigation, armor_skill = self._roll_hit_location_dr(
            target, damage_type
        )
        final_damage = max(0, raw_damage - dr)

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
            reward += 0.05 + 0.02 * final_damage
            self._gain_skill_xp(attacker, skill_name, 2, events)
            if target.hp <= 0 and target.alive:
                target.alive = False
                events.append(f"agent_interact:kill:{target.agent_id}")
                reward += 0.5
                self._gain_skill_xp(attacker, skill_name, 4, events)
        else:
            events.append(f"agent_interact:blocked:{target.agent_id}")
            reward -= 0.01
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
            return -0.01

        raw_damage = self._rng.randint(damage_range[0], damage_range[1])
        raw_damage += self._damage_stat_bonus(attacker, damage_type)
        raw_damage += max(0, (skill_level - 1) // 2)
        dr = self._rng.randint(target.dr_min, max(target.dr_min, target.dr_max))
        final_damage = max(0, raw_damage - dr)

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
        return reward

    def _equipped_weapon(self, agent: AgentState) -> Tuple[str, str, Tuple[int, int], str]:
        for item in reversed(agent.equipped):
            if item in self.weapon_damage_type:
                skill_name = self.weapon_skill_by_item.get(item, "melee")
                return (
                    item,
                    self.weapon_damage_type[item],
                    self.weapon_damage_range[item],
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
        for slot in armor_slots:
            item = target.armor_slots.get(slot)
            if item is None:
                continue
            has_armor_in_slot = True
            armor_mitigation += int(self.item_dr_bonus_vs.get(item, {}).get(damage_type, 0))

        armor_skill = ""
        if has_armor_in_slot:
            armor_skill = HIT_SLOT_TO_ARMOR_SKILL.get(hit_slot, "")
            if armor_skill:
                armor_mitigation += max(0, self._skill_level(target, armor_skill) // 3)

        race = self._race_by_name(target.race_name)
        base_min = int(race.base_dr_min)
        base_max = int(race.base_dr_max)
        if base_max < base_min:
            base_max = base_min
        dr = self._rng.randint(base_min, base_max)
        dr += int(race.dr_bonus_vs.get(damage_type, 0))
        dr += armor_mitigation
        return max(0, dr), hit_slot, max(0, armor_mitigation), armor_skill

    def _roll_armor_dr(self, target: AgentState, damage_type: str) -> int:
        dr, _, _, _ = self._roll_hit_location_dr(target, damage_type)
        return dr

    def _pickup_from_tile(self, agent: AgentState, events: List[str]) -> float:
        assert self.state is not None
        r, c = agent.position
        reward = 0.0

        items = self.state.ground_items.get((r, c), [])
        if items:
            item = items.pop(0)
            agent.inventory.append(item)
            reward += 0.1
            events.append(f"pickup:{item}")
            if item in ("bow", "crossbow"):
                self._gain_skill_xp(agent, "archery", 1, events)
            elif item in ("thrown_rock", "thrown_knife"):
                self._gain_skill_xp(agent, "thrown_weapons", 1, events)
            elif item in ("bandage", "healing_potion", "antidote"):
                self._gain_skill_xp(agent, "medic", 1, events)
            return reward

        chest = self.state.chests.get((r, c))
        if chest and not chest.opened:
            chest.opened = True
            if chest.loot:
                for item in chest.loot:
                    agent.inventory.append(item)
                    events.append(f"chest_open:{item}")
                reward += 0.16 + 0.06 * min(4, len(chest.loot))
            else:
                events.append("chest_open:empty")
                reward -= 0.01
            return reward

        tile_id = self.state.grid[r][c]
        tile = self.tiles[tile_id]
        used = self.state.tile_interactions.get((r, c), 0)
        if (
            tile_id == "food_cache"
            and tile.loot_table
            and used < max(1, tile.max_interactions)
        ):
            self.state.tile_interactions[(r, c)] = used + 1
            item = self._rng.choice(tile.loot_table)
            agent.inventory.append(item)
            reward += 0.2
            events.append(f"loot:{tile_id}:{item}")
        else:
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

    def _observation_window_dims(self, aid: str) -> Tuple[int, int]:
        assert self.state is not None
        cfg = self.config.agent_observation_config.get(aid, {})
        agent = self.state.agents[aid]
        profile = self._profile_for_agent(aid)
        exploration_bonus = max(0, self._skill_level(agent, "exploration"))
        view_width = int(cfg.get("view_width", profile.view_width)) + exploration_bonus
        view_height = int(cfg.get("view_height", profile.view_height)) + exploration_bonus
        return max(1, view_height), max(1, view_width)

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
            dr = abs(other.position[0] - ar)
            dc = abs(other.position[1] - ac)
            if dr <= half_h and dc <= half_w:
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

        if include_grid:
            obs["local_tiles"] = self._local_view_dims(
                agent.position, height=view_height, width=view_width
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
                "hunger": agent.hunger,
                "position": agent.position,
                "equipped_count": len(agent.equipped),
                "armor_slots": dict(agent.armor_slots),
                "strength": agent.strength,
                "dexterity": agent.dexterity,
                "intellect": agent.intellect,
                "skills": dict(agent.skills),
                "skill_xp": dict(agent.skill_xp),
                "overall_level": self._overall_level(agent),
                "encumbrance_ratio": self._encumbrance_ratio(agent),
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
                "nearby_agents": self._nearby_agent_counts(
                    aid=aid, center=agent.position, height=view_height, width=view_width
                ),
                "nearby_monsters": self._nearby_monster_counts(
                    center=agent.position, height=view_height, width=view_width
                ),
            }
        if include_inventory:
            obs["inventory"] = list(agent.inventory)

        return obs

    def _resolve_profile_name(self, agent_id: str, index: int) -> str:
        if agent_id in self.config.agent_profile_map:
            return self.config.agent_profile_map[agent_id]
        if index == 0:
            return "human"
        if index == 1:
            return "orc"
        return "human"

    def _resolve_class_name(self, agent_id: str, index: int) -> str:
        if agent_id in self.config.agent_class_map:
            return self.config.agent_class_map[agent_id]
        if "wanderer" in self.classes:
            return "wanderer"
        ordered = sorted(self.classes.keys())
        if not ordered:
            raise ValueError("No agent classes are loaded")
        return ordered[index % len(ordered)]

    def _resolve_race_name(self, agent_id: str, index: int) -> str:
        if agent_id in self.config.agent_race_map:
            return self.config.agent_race_map[agent_id]
        if agent_id in self.config.agent_profile_map:
            mapped = self.config.agent_profile_map[agent_id]
            if mapped in self.races:
                return mapped
        if index == 0 and "human" in self.races:
            return "human"
        if index == 1 and "orc" in self.races:
            return "orc"
        ordered = sorted(self.races.keys())
        if not ordered:
            raise ValueError("No agent races are loaded")
        return ordered[index % len(ordered)]

    def _profile_by_name(self, name: str) -> AgentProfile:
        if name not in self.profiles:
            raise ValueError(f"Unknown agent profile: {name}")
        return self.profiles[name]

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

    def _local_view_dims(
        self, center: Tuple[int, int], height: int, width: int
    ) -> List[List[str]]:
        assert self.state is not None
        height = max(1, int(height))
        width = max(1, int(width))
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
        distances: List[int] = []
        for other_id, other in self.state.agents.items():
            if other_id == aid or not other.alive:
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
            weight += float(self.item_weight.get(item, 1.0))
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

    def _nearby_agent_counts(
        self, aid: str, center: Tuple[int, int], height: int, width: int
    ) -> Dict[str, int | None]:
        assert self.state is not None
        cr, cc = center
        start_r = cr - (height // 2)
        start_c = cc - (width // 2)
        end_r = start_r + height - 1
        end_c = start_c + width - 1
        total_visible = 0
        adjacent = 0
        nearest: int | None = None
        for other_id, other in self.state.agents.items():
            if other_id == aid or not other.alive:
                continue
            r, c = other.position
            if r < start_r or r > end_r or c < start_c or c > end_c:
                continue
            total_visible += 1
            dist = self._manhattan(center, other.position)
            if dist == 1:
                adjacent += 1
            if nearest is None or dist < nearest:
                nearest = dist
        return {
            "visible": total_visible,
            "adjacent": adjacent,
            "nearest_distance": nearest,
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

    def _manhattan(self, a: Tuple[int, int], b: Tuple[int, int]) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])


class PettingZooParallelRLRLGym(MultiAgentRLRLGym):
    """Primary env class name to signal PettingZoo Parallel-style usage."""
