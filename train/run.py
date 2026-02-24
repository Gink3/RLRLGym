"""CLI entrypoint for training in-repo."""

from __future__ import annotations

import argparse

from .rllib_trainer import RLlibTrainConfig, RLlibTrainer
from .trainer import MultiAgentTrainer, TrainConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train multi-agent policies in RLRLGym")
    p.add_argument("--backend", choices=["custom", "rllib"], default="rllib")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=120)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="outputs/train")
    p.add_argument("--width", type=int, default=20)
    p.add_argument("--height", type=int, default=12)
    p.add_argument("--agents", type=int, default=2)
    p.add_argument("--networks-path", type=str, default="data/agent_networks.json")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--framework", type=str, default="torch")
    p.add_argument("--num-gpus", type=float, default=0.0)
    p.add_argument("--num-rollout-workers", type=int, default=0)
    p.add_argument("--train-batch-size", type=int, default=4000)
    p.add_argument("--replay-save-every", type=int, default=1000)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.backend == "custom":
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
            replay_save_every=args.replay_save_every,
        )
        trainer = MultiAgentTrainer(config)
        result = trainer.train()
        print("Training complete (custom)")
        print("Aggregate:", result["aggregate"])
        print("Artifacts:", result["artifacts"])
        print("Checkpoint:", result["checkpoint"])
        return

    rllib_cfg = RLlibTrainConfig(
        iterations=args.iterations,
        seed=args.seed,
        output_dir=args.output_dir,
        width=args.width,
        height=args.height,
        n_agents=args.agents,
        max_steps=args.max_steps,
        framework=args.framework,
        num_gpus=args.num_gpus,
        num_rollout_workers=args.num_rollout_workers,
        train_batch_size=args.train_batch_size,
        replay_save_every=args.replay_save_every,
    )
    trainer = RLlibTrainer(rllib_cfg)
    summary = trainer.train()
    print("Training complete (RLlib)")
    print("Summary:", summary)


if __name__ == "__main__":
    main()
