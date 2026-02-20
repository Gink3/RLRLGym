"""Window rendering demo for RLRLGym."""

from __future__ import annotations

import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlrlgym import EnvConfig, PettingZooParallelRLRLGym


def main() -> None:
    env = PettingZooParallelRLRLGym(
        EnvConfig(
            width=24,
            height=14,
            n_agents=2,
            max_steps=60,
            render_enabled=True,
        )
    )

    env.reset(seed=21)
    env.open_render_window(title="RLRLGym Live Demo")

    rng = random.Random(21)
    playback_states = [env.capture_playback_state()]

    for _ in range(35):
        actions = {aid: rng.randint(0, 10) for aid in env.agents}
        _, _, terminations, truncations, _ = env.step(actions)
        playback_states.append(env.capture_playback_state())

        if all(terminations.values()) or all(truncations.values()):
            break

    env.play_frames_in_window(playback_states, title="RLRLGym Playback")
    env.run_render_window()


if __name__ == "__main__":
    main()
