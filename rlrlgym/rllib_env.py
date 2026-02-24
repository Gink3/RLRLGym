"""RLlib adapter environment for RLRLGym."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .env import EnvConfig, PettingZooParallelRLRLGym
from .featurize import observation_vector_size, vectorize_observation


class RLRLGymRLlibEnv(MultiAgentEnv):
    """A minimal RLlib-compatible multi-agent env wrapper."""

    def __init__(self, config: Dict[str, object] | None = None) -> None:
        super().__init__()
        cfg = config or {}
        profile_map = cfg.get("agent_profile_map", {"agent_0": "human", "agent_1": "orc"})
        self.base = PettingZooParallelRLRLGym(
            EnvConfig(
                width=int(cfg.get("width", 20)),
                height=int(cfg.get("height", 12)),
                max_steps=int(cfg.get("max_steps", 120)),
                n_agents=int(cfg.get("n_agents", 2)),
                render_enabled=bool(cfg.get("render_enabled", False)),
                agent_profile_map=dict(profile_map),
            )
        )
        self._obs_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(observation_vector_size(),),
            dtype=np.float32,
        )
        self._action_space = spaces.Discrete(11)
        self.possible_agents = list(self.base.possible_agents)
        self.agents = []
        self.observation_spaces = {aid: self._obs_space for aid in self.possible_agents}
        self.action_spaces = {aid: self._action_space for aid in self.possible_agents}
        self._done_agents: set[str] = set()
        self._episode_counter = 0
        self._replay_save_every = int(cfg.get("replay_save_every", 100))
        out_dir = cfg.get("replay_output_dir", "")
        self._replay_output_dir = Path(out_dir) if out_dir else None
        self._capture_replay = False
        self._replay_states = []
        self._replay_actions = []

    @property
    def observation_space(self):
        # RLlib MultiAgentEnv expects a per-agent mapping on the new API stack.
        return self.observation_spaces

    @property
    def action_space(self):
        # RLlib MultiAgentEnv expects a per-agent mapping on the new API stack.
        return self.action_spaces

    def get_observation_space(self, agent_id):
        return self.observation_spaces[agent_id]

    def get_action_space(self, agent_id):
        return self.action_spaces[agent_id]

    def reset(self, *, seed=None, options=None):
        obs, info = self.base.reset(seed=seed, options=options)
        self._done_agents = set()
        self.agents = list(self.base.agents)
        self._episode_counter += 1
        self._capture_replay = (
            self._replay_output_dir is not None
            and self._replay_save_every > 0
            and self._episode_counter % self._replay_save_every == 0
        )
        self._replay_states = (
            [self.base.capture_playback_state()] if self._capture_replay else []
        )
        self._replay_actions = []
        vec_obs = {
            aid: np.array(vectorize_observation(o), dtype=np.float32)
            for aid, o in obs.items()
        }
        return vec_obs, info

    def step(self, action_dict):
        obs, rewards, terminations, truncations, info = self.base.step(action_dict)
        if self._capture_replay:
            self._replay_states.append(self.base.capture_playback_state())
            self._replay_actions.append(
                {aid: int(a) for aid, a in dict(action_dict).items()}
            )
        vec_obs = {}
        out_rewards = {}
        out_infos = {}
        terminateds = {}
        truncateds = {}

        # RLlib must not receive repeated terminal signals for already-done agents.
        for aid in self.possible_agents:
            if aid in self._done_agents:
                continue

            # RLlib expects a final observation for any agent that just got
            # terminated/truncated in this step.
            if aid in obs:
                final_obs = obs[aid]
            elif aid in self.base.state.agents:
                final_obs = self.base._build_observation(aid)  # type: ignore[attr-defined]
            else:
                final_obs = None
            if final_obs is not None:
                vec_obs[aid] = np.array(vectorize_observation(final_obs), dtype=np.float32)
            if aid in rewards:
                out_rewards[aid] = rewards[aid]
            if aid in info:
                out_infos[aid] = info[aid]

            term = bool(terminations.get(aid, False))
            trunc = bool(truncations.get(aid, False))
            if term:
                terminateds[aid] = True
                self._done_agents.add(aid)
            if trunc:
                truncateds[aid] = True
                self._done_agents.add(aid)

        self.agents = [aid for aid in self.base.agents if aid not in self._done_agents]
        all_done = len(self._done_agents) == len(self.possible_agents)
        terminateds["__all__"] = all_done
        truncateds["__all__"] = all_done
        if all_done and self._capture_replay:
            self._write_replay(
                episode=self._episode_counter,
                frames=self._replay_states,
                action_history=self._replay_actions,
            )
            self._capture_replay = False
            self._replay_states = []
            self._replay_actions = []
        return vec_obs, out_rewards, terminateds, truncateds, out_infos

    def _write_replay(
        self, episode: int, frames: list, action_history: list[Dict[str, int]]
    ) -> None:
        if self._replay_output_dir is None:
            return
        out = self._replay_output_dir / "replays"
        out.mkdir(parents=True, exist_ok=True)
        p = out / f"episode_{episode:06d}.replay.json"
        payload = {
            "schema_version": 1,
            "episode": int(episode),
            "frame_count": len(frames),
            "frames": [self._serialize_state(s) for s in frames],
            "actions": [{aid: int(a) for aid, a in x.items()} for x in action_history],
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _serialize_state(self, state) -> Dict[str, object]:
        return {
            "grid": state.grid,
            "tile_interactions": [
                {"position": [r, c], "count": count}
                for (r, c), count in sorted(state.tile_interactions.items())
            ],
            "ground_items": [
                {"position": [r, c], "items": list(items)}
                for (r, c), items in sorted(state.ground_items.items())
            ],
            "agents": {
                aid: {
                    "agent_id": agent.agent_id,
                    "position": [agent.position[0], agent.position[1]],
                    "profile_name": agent.profile_name,
                    "hp": agent.hp,
                    "max_hp": agent.max_hp,
                    "hunger": agent.hunger,
                    "max_hunger": agent.max_hunger,
                    "inventory": list(agent.inventory),
                    "equipped": list(agent.equipped),
                    "alive": agent.alive,
                    "visited": [[r, c] for (r, c) in sorted(agent.visited)],
                    "wait_streak": agent.wait_streak,
                    "recent_positions": [[r, c] for (r, c) in agent.recent_positions],
                }
                for aid, agent in sorted(state.agents.items())
            },
            "step_count": state.step_count,
        }
