"""PettingZoo-style parallel multi-agent roguelike environment."""

from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .constants import (
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
from .mapgen import generate_map, sample_walkable_positions
from .mapgen_config import MapGenConfig, load_mapgen_config
from .models import AgentState, ChestState, EnvState, MonsterState
from .monsters import MonsterDef, MonsterSpawnEntry, load_monster_spawns, load_monsters
from .profiles import AgentProfile, load_profiles
from .races import AgentRace, load_races
from .render import RenderWindow
from .tiles import load_tileset

EDIBLE_ITEMS = {"ration", "fruit"}
MOVE_VALID_REWARD = 0.005
MOVE_STEP_COST = 0.003
MOVE_FOOD_PROGRESS_REWARD = 0.01
MOVE_FOOD_REGRESS_PENALTY = 0.005
EAT_PER_HUNGER_GAIN_REWARD = 0.04
EAT_WASTE_THRESHOLD = 0.8
EAT_WASTE_PENALTY = 0.05
LOW_HUNGER_THRESHOLD = 0.4
LOW_HUNGER_PENALTY_SCALE = 0.01

DAMAGE_TYPE_SLASH = "slash"
DAMAGE_TYPE_PIERCE = "pierce"
DAMAGE_TYPE_BLUNT = "blunt"

WEAPON_DAMAGE_TYPE = {
    "dagger": DAMAGE_TYPE_PIERCE,
    "short_sword": DAMAGE_TYPE_SLASH,
    "long_sword": DAMAGE_TYPE_SLASH,
    "spear": DAMAGE_TYPE_PIERCE,
    "club": DAMAGE_TYPE_BLUNT,
    "mace": DAMAGE_TYPE_BLUNT,
    "bow": DAMAGE_TYPE_PIERCE,
    "crossbow": DAMAGE_TYPE_PIERCE,
    "thrown_rock": DAMAGE_TYPE_BLUNT,
    "thrown_knife": DAMAGE_TYPE_PIERCE,
}
WEAPON_DAMAGE_RANGE = {
    "dagger": (2, 4),
    "short_sword": (3, 5),
    "long_sword": (4, 7),
    "spear": (3, 6),
    "club": (2, 5),
    "mace": (3, 6),
    "bow": (2, 5),
    "crossbow": (3, 6),
    "thrown_rock": (1, 3),
    "thrown_knife": (2, 4),
}
ITEM_DR_BONUS_VS = {
    "shield": {DAMAGE_TYPE_SLASH: 1, DAMAGE_TYPE_PIERCE: 1, DAMAGE_TYPE_BLUNT: 0},
    "leather_armor": {DAMAGE_TYPE_SLASH: 1, DAMAGE_TYPE_PIERCE: 0, DAMAGE_TYPE_BLUNT: 0},
    "chain_mail": {DAMAGE_TYPE_SLASH: 2, DAMAGE_TYPE_PIERCE: 1, DAMAGE_TYPE_BLUNT: -1},
}
UNARMED_DAMAGE_RANGE = (1, 2)

ITEM_WEIGHT = {
    "ration": 1.0,
    "fruit": 0.5,
    "bandage": 0.3,
    "antidote": 0.4,
    "healing_potion": 0.6,
    "dagger": 1.0,
    "short_sword": 1.8,
    "long_sword": 2.4,
    "spear": 2.2,
    "club": 2.0,
    "mace": 2.5,
    "bow": 1.8,
    "crossbow": 2.6,
    "thrown_rock": 0.8,
    "thrown_knife": 0.6,
    "torch": 0.7,
}
CHEST_LOOT_TABLE = [
    "bandage",
    "healing_potion",
    "antidote",
    "dagger",
    "short_sword",
    "long_sword",
    "spear",
    "club",
    "mace",
    "bow",
    "crossbow",
    "thrown_rock",
    "thrown_knife",
    "ration",
    "fruit",
]

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
    monsters_path: str = str(Path("data") / "monsters.json")
    monster_spawns_path: str = str(Path("data") / "monster_spawns.json")
    mapgen_config_path: str = str(Path("data") / "mapgen_config.json")
    agent_observation_config: Dict[str, Dict[str, object]] = field(default_factory=dict)
    agent_profile_map: Dict[str, str] = field(default_factory=dict)
    agent_race_map: Dict[str, str] = field(default_factory=dict)
    agent_class_map: Dict[str, str] = field(default_factory=dict)
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
        self.monsters: Dict[str, MonsterDef] = load_monsters(self.config.monsters_path)
        self.monster_spawns: List[MonsterSpawnEntry] = load_monster_spawns(
            self.config.monster_spawns_path, self.monsters
        )
        self.mapgen_cfg: MapGenConfig = load_mapgen_config(
            self.config.mapgen_config_path
        )
        self._rng = random.Random(0)
        self.possible_agents = [f"agent_{i}" for i in range(self.config.n_agents)]
        self.agents = list(self.possible_agents)
        self.state: Optional[EnvState] = None
        self._last_info: Dict[str, Dict[str, object]] = {}
        self._render_window: Optional[RenderWindow] = None

    def action_space(self, agent_id: str) -> Tuple[int, int]:
        if agent_id not in self.possible_agents:
            raise KeyError(f"Unknown agent: {agent_id}")
        return (0, 10)

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
        self.agents = list(self.possible_agents)
        obs = {aid: self._build_observation(aid) for aid in self.possible_agents}
        info = {
            aid: {
                "action_mask": [1] * 11,
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
            delta_reward, events = self._apply_action(aid, action)
            rewards[aid] += delta_reward
            info[aid]["events"].extend(events)
            self._apply_survival_costs(agent, rewards, aid, info)

            if agent.hp <= 0:
                agent.alive = False
                terminations[aid] = True
                rewards[aid] -= 1.0
                info[aid]["events"].append("death")

        self._apply_monster_turn(rewards, terminations, info)
        self.state.step_count += 1

        if self.state.step_count >= self.config.max_steps:
            for aid in self.possible_agents:
                truncations[aid] = True

        for aid in self.possible_agents:
            profile = self._profile_for_agent(aid)
            rewards[aid] += profile.reward_adjustment(
                events=info[aid]["events"],
                died=terminations[aid],
            )
            agent = self.state.agents[aid]
            info[aid]["alive"] = agent.alive
            info[aid]["profile"] = agent.profile_name
            info[aid]["race"] = agent.race_name
            info[aid]["class"] = agent.class_name
            info[aid]["teammate_distance"] = self._nearest_teammate_distance(aid)

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
        playback_actions: Optional[List[Dict[str, int]]] = None,
    ) -> None:
        if not self.config.render_enabled:
            return
        if self._render_window is None:
            self._render_window = RenderWindow(self.tiles, title=title)
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
                    reward -= max(0.0, 0.03 - 0.003 * athletics_level)
                    events.append("stutter_penalty")
                agent.recent_positions.append(agent.position)
                agent.recent_positions = agent.recent_positions[-5:]
                agent.wait_streak = 0
            else:
                reward -= max(0.0, 0.02 - 0.002 * athletics_level)
                events.append("bump")

        elif action == ACTION_WAIT:
            agent.wait_streak += 1
            reward -= 0.02
            events.append("wait")
            if agent.wait_streak > 3:
                reward -= 0.03
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
                agent.equipped.append(item)
                reward += 0.08
                events.append(f"equip:{item}")
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
                break
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
        dr = self._roll_armor_dr(target, damage_type)
        final_damage = max(0, raw_damage - dr)

        events.append(
            f"agent_interact:attack:{target.agent_id}:{weapon}:{damage_type}"
        )
        events.append(
            f"agent_interact:attack_roll:{raw_damage}:dr:{dr}:final:{final_damage}"
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
            if item in WEAPON_DAMAGE_TYPE:
                if item in ("bow", "crossbow"):
                    skill_name = "archery"
                elif item in ("thrown_rock", "thrown_knife"):
                    skill_name = "thrown_weapons"
                else:
                    skill_name = "melee"
                return (
                    item,
                    WEAPON_DAMAGE_TYPE[item],
                    WEAPON_DAMAGE_RANGE[item],
                    skill_name,
                )
        return "unarmed", DAMAGE_TYPE_BLUNT, UNARMED_DAMAGE_RANGE, "melee"

    def _roll_armor_dr(self, target: AgentState, damage_type: str) -> int:
        race = self._race_by_name(target.race_name)
        base_min = int(race.base_dr_min)
        base_max = int(race.base_dr_max)
        if base_max < base_min:
            base_max = base_min
        dr = self._rng.randint(base_min, base_max)
        dr += int(race.dr_bonus_vs.get(damage_type, 0))
        for item in target.equipped:
            dr += int(ITEM_DR_BONUS_VS.get(item, {}).get(damage_type, 0))
        return max(0, dr)

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
    ) -> None:
        agent.hunger = max(0, agent.hunger - 1)
        if agent.hunger == 0:
            agent.hp -= 1
            rewards[aid] -= 0.05
            info[aid]["events"].append("starve_tick")
        hunger_ratio = agent.hunger / max(1, agent.max_hunger)
        if hunger_ratio < LOW_HUNGER_THRESHOLD:
            pressure = (LOW_HUNGER_THRESHOLD - hunger_ratio) / LOW_HUNGER_THRESHOLD
            rewards[aid] -= LOW_HUNGER_PENALTY_SCALE * pressure
            info[aid]["events"].append("low_hunger_pressure")

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
        exploration_bonus = max(0, self._skill_level(agent, "exploration"))
        view_width = int(cfg.get("view_width", profile.view_width)) + exploration_bonus
        view_height = int(cfg.get("view_height", profile.view_height)) + exploration_bonus
        view_width = max(1, view_width)
        view_height = max(1, view_height)
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
            obs["stats"] = {
                "hp": agent.hp,
                "hunger": agent.hunger,
                "position": agent.position,
                "equipped_count": len(agent.equipped),
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
                "nearby_chests": self._nearby_chest_counts(
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
            if any(item in EDIBLE_ITEMS for item in items):
                dist = abs(row - r) + abs(col - c)
                if best is None or dist < best:
                    best = dist
        for (r, c), chest in self.state.chests.items():
            if chest.opened:
                continue
            if any(item in EDIBLE_ITEMS for item in chest.loot):
                dist = abs(row - r) + abs(col - c)
                if best is None or dist < best:
                    best = dist

        for r, grid_row in enumerate(self.state.grid):
            for c, tile_id in enumerate(grid_row):
                tile = self.tiles[tile_id]
                if not tile.loot_table:
                    continue
                if not any(item in EDIBLE_ITEMS for item in tile.loot_table):
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
            loot = [self._rng.choice(CHEST_LOOT_TABLE) for _ in range(loot_count)]
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
        leveled = False
        level = self._skill_level(agent, skill)
        while current >= self._skill_xp_to_next(level):
            current -= self._skill_xp_to_next(level)
            level += 1
            leveled = True
        agent.skill_xp[skill] = current
        agent.skills[skill] = level
        if leveled:
            events.append(f"skill_up:{skill}:{level}")

    def _skill_xp_to_next(self, level: int) -> int:
        return 20 + 15 * max(0, int(level))

    def _overall_level(self, agent: AgentState) -> int:
        return int(sum(max(0, int(v)) for v in agent.skills.values()))

    def _carried_weight(self, agent: AgentState) -> float:
        weight = 0.0
        for item in agent.inventory + agent.equipped:
            weight += float(ITEM_WEIGHT.get(item, 1.0))
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

            target_id = self._nearest_alive_agent_id(monster.position)
            if target_id is None:
                return
            target = self.state.agents[target_id]
            dist = self._manhattan(monster.position, target.position)
            if dist == 1:
                self._monster_attack(monster, target, rewards, terminations, info)
            else:
                self._monster_move_toward_target(monster, target.position, info)

    def _nearest_alive_agent_id(self, position: Tuple[int, int]) -> str | None:
        assert self.state is not None
        best_id: str | None = None
        best_dist: int | None = None
        for aid in self.possible_agents:
            agent = self.state.agents[aid]
            if not agent.alive or agent.hp <= 0:
                continue
            dist = self._manhattan(position, agent.position)
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
        dr = self._roll_armor_dr(target, DAMAGE_TYPE_BLUNT)
        final_damage = max(0, raw_damage - dr)
        target_events.append(
            f"monster_hit_roll:{monster.monster_id}:raw:{raw_damage}:dr:{dr}:final:{final_damage}"
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
