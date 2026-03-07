"""Simple synchronous vectorized environment wrapper."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence


class SyncVectorEnv:
    def __init__(self, envs: Sequence[Any]) -> None:
        if not envs:
            raise ValueError("Need at least one environment")
        self.envs = list(envs)

    def reset(self, seed: int | None = None):
        out = []
        for i, env in enumerate(self.envs):
            local_seed = None if seed is None else seed + i
            out.append(env.reset(seed=local_seed))
        return out

    def step(self, batched_actions: List[Dict[str, int]]):
        if len(batched_actions) != len(self.envs):
            raise ValueError("Actions batch size must match number of envs")
        out = []
        for env, actions in zip(self.envs, batched_actions):
            out.append(env.step(actions))
        return out
