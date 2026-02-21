"""CLI entrypoint for training in-repo."""

from __future__ import annotations

import argparse

from .trainer import MultiAgentTrainer, TrainConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train neural multi-agent policies in RLRLGym")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="outputs/train")
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--height", type=int, default=12)
    p.add_argument("--agents", type=int, default=2)
    p.add_argument("--networks-path", type=str, default="data/agent_networks.json")
    return p


def main() -> None:
    args = build_parser().parse_args()
    config = TrainConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        n_agents=args.agents,
        render_enabled=False,
        networks_path=args.networks_path,
    )
    trainer = MultiAgentTrainer(config)
    result = trainer.train()
    print("Training complete")
    print("Aggregate:", result["aggregate"])
    print("Artifacts:", result["artifacts"])
    print("Checkpoint:", result["checkpoint"])


if __name__ == "__main__":
    main()
