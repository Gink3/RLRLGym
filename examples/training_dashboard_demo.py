"""Generate training metrics and a dashboard report."""

from __future__ import annotations

import random
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlrlgym import EnvConfig, PettingZooParallelRLRLGym, TrainingLogger


def run_episode(env: PettingZooParallelRLRLGym, logger: TrainingLogger, seed: int) -> None:
    env.reset(seed=seed)
    logger.start_episode(env.possible_agents)
    rng = random.Random(seed)

    for _ in range(env.config.max_steps):
        actions = {aid: rng.randint(0, 10) for aid in env.agents}
        _, rewards, terminations, truncations, info = env.step(actions)
        logger.log_step(rewards, terminations, truncations, info, actions=actions)

        if not env.agents:
            break

    alive = {aid: env.state.agents[aid].alive for aid in env.possible_agents}
    logger.end_episode(step_count=env.state.step_count, alive_agents=alive)


def main() -> None:
    env = PettingZooParallelRLRLGym(
        EnvConfig(
            width=20,
            height=12,
            n_agents=2,
            max_steps=80,
            render_enabled=False,
            agent_profile_map={"agent_0": "human", "agent_1": "orc"},
        )
    )
    logger = TrainingLogger(output_dir="outputs")

    for episode_seed in range(10):
        run_episode(env, logger, seed=episode_seed)

    paths = logger.write_outputs()
    aggregate = logger.aggregate_metrics()
    print("Training run complete")
    print("Aggregate:", aggregate)
    print("Artifacts:", paths)


if __name__ == "__main__":
    main()
