"""Multi-agent roguelike RL gym environment."""

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
from .render import AsciiRenderer, RenderWindow
from .tiles import load_tileset


@dataclass
class EnvConfig:
    width: int = 24
    height: int = 16
    max_steps: int = 150
    n_agents: int = 2
    tiles_path: str = str(Path("data") / "tiles.json")
    agent_observation_config: Dict[str, Dict[str, object]] = field(default_factory=dict)
    render_enabled: bool = True


class MultiAgentRLRLGym:
    """Gym-like API with multi-agent step signature.

    `reset(seed)` -> observations, info
    `step(actions)` -> observations, rewards, terminations, truncations, info
    """

    metadata = {"name": "RLRLGym-v0", "render_modes": ["ansi", "window"]}

    def __init__(self, config: Optional[EnvConfig] = None) -> None:
        self.config = config or EnvConfig()
        self.tiles = load_tileset(self.config.tiles_path)
        self.renderer = AsciiRenderer(self.tiles)
        self._rng = random.Random(0)
        self.possible_agents = [f"agent_{i}" for i in range(self.config.n_agents)]
        self.state: Optional[EnvState] = None
        self._last_info: Dict[str, Dict[str, object]] = {}
        self._render_window: Optional[RenderWindow] = None

    @property
    def action_space(self) -> Dict[str, Tuple[int, int]]:
        return {agent_id: (0, 10) for agent_id in self.possible_agents}

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng.seed(seed)

        grid = generate_map(self.config.width, self.config.height, self.tiles, self._rng)
        starts = sample_walkable_positions(grid, self.tiles, self.config.n_agents, self._rng)

        agents: Dict[str, AgentState] = {}
        for i, agent_id in enumerate(self.possible_agents):
            pos = starts[i]
            agent = AgentState(agent_id=agent_id, position=pos)
            agent.visited.add(pos)
            agents[agent_id] = agent

        self.state = EnvState(grid=grid, tile_interactions={}, ground_items={}, agents=agents, step_count=0)
        obs = {aid: self._build_observation(aid) for aid in self.possible_agents}
        info = {aid: {"action_mask": [1] * 11, "alive": True} for aid in self.possible_agents}
        self._last_info = info
        if self.config.render_enabled and self._render_window is not None:
            self._render_window.update_state(self.state, focus_choices=self.possible_agents)
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

        self.state.step_count += 1
        if self.state.step_count >= self.config.max_steps:
            for aid in self.possible_agents:
                truncations[aid] = True

        observations = {
            aid: self._build_observation(aid)
            for aid in self.possible_agents
            if self.state.agents[aid].alive or truncations[aid]
        }
        self._last_info = info
        if self.config.render_enabled and self._render_window is not None:
            self._render_window.update_state(self.state, focus_choices=self.possible_agents)
        return observations, rewards, terminations, truncations, info

    def render(self, focus_agent: Optional[str] = None, zoom: int = 0, color: bool = True) -> str:
        if not self.config.render_enabled:
            return ""
        if self.state is None:
            return ""
        return self.renderer.render(self.state, focus_agent=focus_agent, zoom=zoom, color=color)

    def open_render_window(self, title: str = "RLRLGym Viewer") -> None:
        if self._render_window is None:
            self._render_window = RenderWindow(self.tiles, title=title)
        if self.state is not None:
            self._render_window.update_state(self.state, focus_choices=self.possible_agents)

    def close_render_window(self) -> None:
        if self._render_window is not None:
            self._render_window.close()
            self._render_window = None

    def play_frames_in_window(self, frames: List[str], title: str = "RLRLGym Playback") -> None:
        if self._render_window is None:
            self._render_window = RenderWindow(self.tiles, title=title)
        self._render_window.set_frames(frames)
        self._render_window.play()

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
            dr, dc = MOVE_DELTAS[action]
            nr, nc = agent.position[0] + dr, agent.position[1] + dc
            if self._walkable(nr, nc):
                old_pos = agent.position
                agent.position = (nr, nc)
                events.append(f"move:{old_pos}->{agent.position}")
                if agent.position not in agent.visited:
                    reward += 0.05
                    events.append("explore")
                    agent.visited.add(agent.position)
                # Penalize immediate backtracking loops.
                if len(agent.recent_positions) >= 2 and agent.recent_positions[-2] == agent.position:
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
            if "ration" in agent.inventory:
                agent.inventory.remove("ration")
                agent.hunger = min(20, agent.hunger + 8)
                reward += 0.25
                events.append("eat:ration")
            elif "fruit" in agent.inventory:
                agent.inventory.remove("fruit")
                agent.hunger = min(20, agent.hunger + 4)
                reward += 0.15
                events.append("eat:fruit")
            else:
                reward -= 0.02
                events.append("eat_fail")

        elif action == ACTION_EQUIP:
            if agent.inventory:
                item = agent.inventory.pop(0)
                agent.equipped.append(item)
                reward += 0.08
                events.append(f"equip:{item}")
            else:
                reward -= 0.01

        elif action == ACTION_USE:
            if "bandage" in agent.inventory and agent.hp < 10:
                agent.inventory.remove("bandage")
                agent.hp = min(10, agent.hp + 3)
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
                actor.hp = min(10, actor.hp + 1)
                reward += 0.1
                events.append("interact:shrine")
            elif tile_id == "water":
                actor.hunger = min(20, actor.hunger + 1)
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
            if abs(other.position[0] - actor.position[0]) + abs(other.position[1] - actor.position[1]) == 1:
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
        radius = int(cfg.get("view_radius", 2))
        include_grid = bool(cfg.get("include_grid", True))
        include_stats = bool(cfg.get("include_stats", True))
        include_inventory = bool(cfg.get("include_inventory", True))

        agent = self.state.agents[aid]
        obs: Dict[str, object] = {"step": self.state.step_count, "alive": agent.alive}

        if include_grid:
            obs["local_tiles"] = self._local_view(agent.position, radius)
        if include_stats:
            obs["stats"] = {
                "hp": agent.hp,
                "hunger": agent.hunger,
                "position": agent.position,
                "equipped_count": len(agent.equipped),
            }
        if include_inventory:
            obs["inventory"] = list(agent.inventory)

        return obs

    def _local_view(self, center: Tuple[int, int], radius: int) -> List[List[str]]:
        assert self.state is not None
        cr, cc = center
        view: List[List[str]] = []
        for r in range(cr - radius, cr + radius + 1):
            row: List[str] = []
            for c in range(cc - radius, cc + radius + 1):
                if r < 0 or c < 0 or r >= len(self.state.grid) or c >= len(self.state.grid[0]):
                    row.append("void")
                else:
                    tile_id = self.state.grid[r][c]
                    row.append(tile_id)
            view.append(row)
        return view
