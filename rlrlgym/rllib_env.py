"""RLlib adapter environment for RLRLGym."""

from __future__ import annotations

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
        vec_obs = {
            aid: np.array(vectorize_observation(o), dtype=np.float32)
            for aid, o in obs.items()
        }
        return vec_obs, info

    def step(self, action_dict):
        obs, rewards, terminations, truncations, info = self.base.step(action_dict)
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
        return vec_obs, out_rewards, terminateds, truncateds, out_infos
