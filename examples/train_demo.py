"""Run a short in-repo training demo using the train module."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.trainer import MultiAgentTrainer, TrainConfig


def main() -> None:
    trainer = MultiAgentTrainer(
        TrainConfig(
            episodes=15,
            max_steps=80,
            seed=7,
            output_dir="outputs/train_demo",
            width=20,
            height=12,
            n_agents=2,
        )
    )
    result = trainer.train()
    print("Training complete")
    print("Aggregate:", result["aggregate"])
    print("Checkpoint:", result["checkpoint"])
    print("Artifacts:", result["artifacts"])


if __name__ == "__main__":
    main()
