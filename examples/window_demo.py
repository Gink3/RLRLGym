"""Window rendering demo for RLRLGym.

Runs a short rollout, updates a dedicated render window live, then plays back
captured frames with pause/play/fast-forward controls.
"""

from __future__ import annotations

import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlrlgym import EnvConfig, MultiAgentRLRLGym


def main() -> None:
    env = MultiAgentRLRLGym(
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
    frames = [env.render(color=False)]

    # Live rollout with window updates on each step.
    for _ in range(35):
        actions = {aid: rng.randint(0, 10) for aid in env.possible_agents}
        _, _, terminations, truncations, _ = env.step(actions)
        frames.append(env.render(color=False))

        if all(terminations.values()) or all(truncations.values()):
            break

    # Switch to playback mode in the same window.
    env.play_frames_in_window(frames, title="RLRLGym Playback")
    env.run_render_window()


if __name__ == "__main__":
    main()
