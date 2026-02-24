"""PettingZoo-style parallel multi-agent roguelike environment."""

from __future__ import annotations

import copy
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
from .mapgen import generate_map, sample_walkable_positions
from .models import AgentState, EnvState
from .profiles import AgentProfile, load_profiles
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


@dataclass
class EnvConfig:
    width: int = 50
    height: int = 50
    max_steps: int = 150
    n_agents: int = 2
    tiles_path: str = str(Path("data") / "tiles.json")
    profiles_path: str = str(Path("data") / "agent_profiles.json")
    agent_observation_config: Dict[str, Dict[str, object]] = field(default_factory=dict)
    agent_profile_map: Dict[str, str] = field(default_factory=dict)
    render_enabled: bool = True


class MultiAgentRLRLGym:
    """PettingZoo Parallel-like API for multi-agent training.

    `reset(seed, options)` -> observations, info
    `step(actions)` -> observations, rewards, terminations, truncations, info
    """

    metadata = {"name": "RLRLGym-v0", "render_modes": ["window"]}

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self.config = config or EnvConfig()
        self.tiles = load_tileset(self.config.tiles_path)
        self.profiles: Dict[str, AgentProfile] = load_profiles(
            self.config.profiles_path
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
        keys = ["step", "alive", "profile"]
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
            self.config.width, self.config.height, self.tiles, self._rng
        )
        starts = sample_walkable_positions(
            grid, self.tiles, self.config.n_agents, self._rng
        )

        agents: Dict[str, AgentState] = {}
        for i, agent_id in enumerate(self.possible_agents):
            pos = starts[i]
            profile_name = self._resolve_profile_name(agent_id, i)
            profile = self._profile_by_name(profile_name)
            agent = AgentState(
                agent_id=agent_id,
                position=pos,
                profile_name=profile.name,
                hp=profile.max_hp,
                max_hp=profile.max_hp,
                hunger=profile.max_hunger,
                max_hunger=profile.max_hunger,
            )
            agent.visited.add(pos)
            agents[agent_id] = agent

        self.state = EnvState(
            grid=grid,
            tile_interactions={},
            ground_items={},
            agents=agents,
            step_count=0,
        )
        self.agents = list(self.possible_agents)
        obs = {aid: self._build_observation(aid) for aid in self.possible_agents}
        info = {
            aid: {
                "action_mask": [1] * 11,
                "alive": True,
                "profile": self.state.agents[aid].profile_name,
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
            if not agent.alive:
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

            profile = self._profile_for_agent(aid)
            rewards[aid] += profile.reward_adjustment(
                events=info[aid]["events"],
                died=terminations[aid],
            )
            info[aid]["alive"] = agent.alive
            info[aid]["profile"] = agent.profile_name
            info[aid]["teammate_distance"] = self._nearest_teammate_distance(aid)

        self.state.step_count += 1
        if self.state.step_count >= self.config.max_steps:
            for aid in self.possible_agents:
                truncations[aid] = True

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
            dr, dc = MOVE_DELTAS[action]
            nr, nc = agent.position[0] + dr, agent.position[1] + dc
            if self._walkable(nr, nc):
                old_pos = agent.position
                agent.position = (nr, nc)
                events.append(f"move:{old_pos}->{agent.position}")
                reward += MOVE_VALID_REWARD
                reward -= MOVE_STEP_COST
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
                    reward -= 0.03
                    events.append("stutter_penalty")
                agent.recent_positions.append(agent.position)
                agent.recent_positions = agent.recent_positions[-5:]
                agent.wait_streak = 0
            else:
                reward -= 0.02
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
            if "bandage" in agent.inventory and agent.hp < agent.max_hp:
                agent.inventory.remove("bandage")
                agent.hp = min(agent.max_hp, agent.hp + 3)
                reward += 0.12
                events.append("use:bandage")
            else:
                reward -= 0.01

        elif action == ACTION_INTERACT:
            reward += self._interact(agent, aid, events)

        return reward, events

    def _interact(self, actor: AgentState, actor_id: str, events: List[str]) -> float:
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

        for other_id, other in self.state.agents.items():
            if other_id == actor_id or not other.alive:
                continue
            if (
                abs(other.position[0] - actor.position[0])
                + abs(other.position[1] - actor.position[1])
                == 1
            ):
                reward += 0.06
                events.append(f"agent_interact:{other_id}")
                break
        return reward

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
            return reward

        tile_id = self.state.grid[r][c]
        tile = self.tiles[tile_id]
        used = self.state.tile_interactions.get((r, c), 0)
        if tile.loot_table and used < max(1, tile.max_interactions):
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
        return True

    def _build_observation(self, aid: str) -> Dict[str, object]:
        assert self.state is not None
        cfg = self.config.agent_observation_config.get(aid, {})

        agent = self.state.agents[aid]
        profile = self._profile_for_agent(aid)
        view_width = int(cfg.get("view_width", profile.view_width))
        view_height = int(cfg.get("view_height", profile.view_height))
        include_grid = bool(cfg.get("include_grid", profile.include_grid))
        include_stats = bool(cfg.get("include_stats", profile.include_stats))
        include_inventory = bool(
            cfg.get("include_inventory", profile.include_inventory)
        )
        obs: Dict[str, object] = {"step": self.state.step_count, "alive": agent.alive}
        obs["profile"] = profile.name

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
                "nearby_item_counts": nearby_item_counts,
                "tile_interaction_counts": self._tile_interaction_counts(
                    agent.position
                ),
                "teammate_distance": self._nearest_teammate_distance(aid),
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

    def _profile_by_name(self, name: str) -> AgentProfile:
        if name not in self.profiles:
            raise ValueError(f"Unknown agent profile: {name}")
        return self.profiles[name]

    def _profile_for_agent(self, agent_id: str) -> AgentProfile:
        assert self.state is not None
        return self._profile_by_name(self.state.agents[agent_id].profile_name)

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


class PettingZooParallelRLRLGym(MultiAgentRLRLGym):
    """Primary env class name to signal PettingZoo Parallel-style usage."""
