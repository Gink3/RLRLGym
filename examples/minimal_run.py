"""Minimal running example for RLRLGym."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlrlgym import EnvConfig, MultiAgentRLRLGym, PlaybackController
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
        agent_observation_config={
            "agent_0": {"view_radius": 2, "include_inventory": True},
            "agent_1": {"view_radius": 1, "include_inventory": False},
        },
    )
    env = MultiAgentRLRLGym(config)
    observations, info = env.reset(seed=7)

    print("Reset complete")
    print("Initial obs keys:", list(observations.keys()))
    print(env.render(color=False))

    frames = [env.render(color=False)]
    for t in range(8):
        actions = {"agent_0": scripted_policy(t), "agent_1": scripted_policy(t + 2)}
        observations, rewards, terminations, truncations, info = env.step(actions)
        frames.append(env.render(color=False))
        print(f"\\nStep {t + 1}")
        print("Actions:", actions)
        print("Rewards:", rewards)
        print("Terminations:", terminations)
        print("Truncations:", truncations)

    playback = PlaybackController(frames=frames)
    playback.fast_forward(3.0)
    print("\\nPlayback preview (first 3 frames at fast-forward speed):")
    for frame in playback.run(limit=3):
        print(frame)
        print("-" * 30)


if __name__ == "__main__":
    main()
