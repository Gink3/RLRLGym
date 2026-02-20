"""Minimal running example for RLRLGym (headless training loop)."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlrlgym import EnvConfig, PettingZooParallelRLRLGym
from rlrlgym.constants import (
    ACTION_EAT,
    ACTION_INTERACT,
    ACTION_LOOT,
    ACTION_MOVE_EAST,
    ACTION_MOVE_NORTH,
    ACTION_MOVE_SOUTH,
    ACTION_MOVE_WEST,
    ACTION_WAIT,
)


def scripted_policy(step: int) -> int:
    script = [
        ACTION_MOVE_EAST,
        ACTION_MOVE_SOUTH,
        ACTION_LOOT,
        ACTION_EAT,
        ACTION_INTERACT,
        ACTION_MOVE_WEST,
        ACTION_MOVE_NORTH,
        ACTION_WAIT,
    ]
    return script[step % len(script)]


def main() -> None:
    config = EnvConfig(
        width=18,
        height=12,
        max_steps=20,
        n_agents=2,
        render_enabled=False,
        agent_observation_config={
            "agent_0": {"view_radius": 2, "include_inventory": True},
            "agent_1": {"view_radius": 1, "include_inventory": False},
        },
    )
    env = PettingZooParallelRLRLGym(config)
    observations, _ = env.reset(seed=7)

    print("Reset complete")
    print("Active agents:", env.agents)
    print("Initial obs keys:", list(observations.keys()))

    for t in range(8):
        actions = {"agent_0": scripted_policy(t), "agent_1": scripted_policy(t + 2)}
        observations, rewards, terminations, truncations, _ = env.step(actions)
        print(
            f"Step {t + 1}: rewards={rewards} terminations={terminations} truncations={truncations}"
        )

        if not env.agents:
            print("All agents done")
            break


if __name__ == "__main__":
    main()
