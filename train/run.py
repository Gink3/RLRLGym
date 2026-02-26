"""CLI entrypoint for training in-repo."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from .rllib_trainer import RLlibTrainConfig, RLlibTrainer
from .trainer import MultiAgentTrainer, TrainConfig


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train multi-agent policies in RLRLGym")
    p.add_argument("--backend", choices=["custom", "rllib"], default="rllib")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=str, default="outputs/train")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--agents", type=int, default=None)
    p.add_argument("--networks-path", type=str, default="data/agent_networks.json")
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--framework", type=str, default="torch")
    p.add_argument("--num-gpus", type=float, default=0.0)
    p.add_argument("--num-rollout-workers", type=int, default=0)
    p.add_argument("--train-batch-size", type=int, default=4000)
    p.add_argument("--replay-save-every", type=int, default=5000)
    p.add_argument("--env-config-path", type=str, default="data/env_config.json")
    p.add_argument("--curriculum-path", type=str, default="data/curriculum_phases.json")
    p.add_argument(
        "--shared-policy",
        action="store_true",
        help="Use one shared policy for all agents (stabilizes early training).",
    )
    p.add_argument(
        "--no-curriculum",
        action="store_true",
        help="Disable two-phase curriculum scheduling in RLlib backend.",
    )
    p.add_argument(
        "--no-aim",
        action="store_true",
        help="Disable Aim metric logging.",
    )
    p.add_argument(
        "--aim-experiment",
        type=str,
        default="rlrlgym",
        help="Aim experiment name.",
    )
    return p


def _timestamped_output_dir(raw_output_dir: str) -> str:
    ts = datetime.now().strftime("%Y%m%d-%H%M")
    out = Path(raw_output_dir)
    parts = out.parts
    if parts and parts[0] == "outputs":
        remainder = list(parts[1:])
        if remainder and remainder[0] == "train":
            remainder = remainder[1:]
        suffix = Path(*remainder) if remainder else Path()
        return str(Path("outputs") / ts / suffix)
    return str(out / ts)


def main() -> None:
    args = build_parser().parse_args()
    resolved_output_dir = _timestamped_output_dir(args.output_dir)
    print(f"Output path: {Path(resolved_output_dir).resolve()}")
    if args.backend == "custom":
        config = TrainConfig(
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            output_dir=resolved_output_dir,
            width=args.width,
            height=args.height,
            n_agents=args.agents,
            render_enabled=False,
            networks_path=args.networks_path,
            replay_save_every=args.replay_save_every,
            env_config_path=args.env_config_path,
            aim_enabled=not bool(args.no_aim),
            aim_experiment=args.aim_experiment,
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
        output_dir=resolved_output_dir,
        width=args.width,
        height=args.height,
        n_agents=args.agents,
        max_steps=args.max_steps,
        framework=args.framework,
        num_gpus=args.num_gpus,
        num_rollout_workers=args.num_rollout_workers,
        train_batch_size=args.train_batch_size,
        replay_save_every=args.replay_save_every,
        env_config_path=args.env_config_path,
        curriculum_path=args.curriculum_path,
        shared_policy=bool(args.shared_policy),
        curriculum_enabled=not bool(args.no_curriculum),
        aim_enabled=not bool(args.no_aim),
        aim_experiment=args.aim_experiment,
    )
    trainer = RLlibTrainer(rllib_cfg)
    summary = trainer.train()
    print("Training complete (RLlib)")
    print("Summary:", summary)


if __name__ == "__main__":
    main()
