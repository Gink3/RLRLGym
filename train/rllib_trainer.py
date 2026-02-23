"""RLlib-based trainer for RLRLGym."""

from __future__ import annotations

import json
import logging
import numbers
import os
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass
class RLlibTrainConfig:
    iterations: int = 100
    seed: int = 0
    output_dir: str = "outputs/train_rllib"
    width: int = 20
    height: int = 12
    n_agents: int = 2
    max_steps: int = 120
    framework: str = "torch"
    num_gpus: int = 0
    num_rollout_workers: int = 0
    train_batch_size: int = 4000


class RLlibTrainer:
    def __init__(self, config: RLlibTrainConfig) -> None:
        self.config = config
        # Must be set before importing Ray to affect logger callback defaults.
        os.environ.setdefault("TUNE_DISABLE_AUTO_CALLBACK_LOGGERS", "1")
        os.environ.setdefault("RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO", "0")
        os.environ.setdefault("RAY_AIR_NEW_OUTPUT", "1")
        try:
            import ray
            from ray import air
            from ray.rllib.algorithms.callbacks import DefaultCallbacks
            from ray.rllib.algorithms.ppo import PPOConfig
            from ray.tune.registry import register_env
            from rlrlgym.rllib_env import RLRLGymRLlibEnv
        except Exception as exc:  # pragma: no cover - dependency/runtime specific
            raise RuntimeError(
                "RLlib backend requires dependencies. Install with: "
                "pip install 'ray[rllib]' numpy gymnasium"
            ) from exc

        self._ray = ray
        self._air = air
        self._DefaultCallbacks = DefaultCallbacks
        self._PPOConfig = PPOConfig
        self._register_env = register_env
        self._RLRLGymRLlibEnv = RLRLGymRLlibEnv
        self._configure_ray_storage_defaults()
        # Reduce noisy transitional warnings emitted through logging channels.
        logging.getLogger("ray.rllib.algorithms.algorithm_config").setLevel(logging.ERROR)
        logging.getLogger("ray.tune.logger.unified").setLevel(logging.ERROR)
        logging.getLogger("ray.rllib.utils.deprecation").setLevel(logging.ERROR)
        logging.getLogger("ray.rllib.utils.sgd").setLevel(logging.ERROR)

    def _configure_ray_storage_defaults(self) -> None:
        """Force Ray result storage into workspace-writable path."""
        storage_root = (Path(self.config.output_dir) / "ray_results").resolve()
        storage_root.mkdir(parents=True, exist_ok=True)
        try:
            import ray.rllib.algorithms.algorithm as algo_mod
            import ray.train.constants as train_constants

            train_constants.DEFAULT_STORAGE_PATH = storage_root.as_posix()
            algo_mod.DEFAULT_STORAGE_PATH = storage_root.as_posix()
        except Exception:
            # Non-fatal; RLlib will use its defaults if patching fails.
            pass

    def train(self) -> Dict[str, object]:
        env_name = "RLRLGymRLlib-v0"
        window = 50
        recent_returns: deque[float] = deque(maxlen=window)
        recent_wins: deque[float] = deque(maxlen=window)
        recent_survival: deque[float] = deque(maxlen=window)
        recent_starvation: deque[float] = deque(maxlen=window)
        recent_loss: deque[float] = deque(maxlen=window)
        recent_teammate_dist: deque[float] = deque(maxlen=window)

        env_config = {
            "width": self.config.width,
            "height": self.config.height,
            "max_steps": self.config.max_steps,
            "n_agents": self.config.n_agents,
            "render_enabled": False,
            "agent_profile_map": {"agent_0": "human", "agent_1": "orc"},
        }

        self._register_env(env_name, lambda cfg: self._RLRLGymRLlibEnv(cfg))
        probe_env = self._RLRLGymRLlibEnv(env_config)
        sample_obs_space = probe_env.observation_spaces["agent_0"]
        sample_action_space = probe_env.action_spaces["agent_0"]

        class MetricsCallbacks(self._DefaultCallbacks):
            def on_episode_end(self, *, episode, **kwargs):
                agent_ids = []
                if hasattr(episode, "get_agents"):
                    agent_ids = list(episode.get_agents())
                elif hasattr(episode, "_agent_to_last_info"):
                    agent_ids = list(episode._agent_to_last_info.keys())

                alive_flags = []
                starvation_flags = []
                teammate_dists = []
                for aid in agent_ids:
                    info = episode.last_info_for(aid)
                    if not info:
                        continue
                    alive = bool(info.get("alive", False))
                    events = info.get("events", [])
                    starved = bool("starve_tick" in events and "death" in events)
                    td = info.get("teammate_distance")
                    alive_flags.append(1.0 if alive else 0.0)
                    starvation_flags.append(1.0 if starved else 0.0)
                    if td is not None:
                        teammate_dists.append(float(td))

                if alive_flags:
                    episode.custom_metrics["win_rate"] = 1.0 if any(v > 0 for v in alive_flags) else 0.0
                if starvation_flags:
                    episode.custom_metrics["starvation_rate"] = sum(starvation_flags) / len(starvation_flags)
                if teammate_dists:
                    episode.custom_metrics["mean_teammate_distance"] = sum(teammate_dists) / len(teammate_dists)

        # Reduce warning noise from known Ray transitional deprecations.
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"ray\..*")
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"ray\..*")

        policy_ids = ["human_policy", "orc_policy"]

        def policy_mapping_fn(agent_id, *args, **kwargs):
            return "human_policy" if agent_id == "agent_0" else "orc_policy"

        ppo = (
            self._PPOConfig()
            .environment(env=env_name, env_config=env_config)
            .api_stack(
                enable_rl_module_and_learner=True,
                enable_env_runner_and_connector_v2=True,
            )
            .framework(self.config.framework)
            .resources(num_gpus=self.config.num_gpus)
            .env_runners(num_env_runners=self.config.num_rollout_workers)
            .training(train_batch_size=self.config.train_batch_size)
            .callbacks(MetricsCallbacks)
            .multi_agent(
                policies={
                    "human_policy": (
                        None,
                        sample_obs_space,
                        sample_action_space,
                        {},
                    ),
                    "orc_policy": (
                        None,
                        sample_obs_space,
                        sample_action_space,
                        {},
                    ),
                },
                policy_mapping_fn=policy_mapping_fn,
                policies_to_train=policy_ids,
            )
            .debugging(seed=self.config.seed, log_sys_usage=False)
        )

        algo = ppo.build_algo()

        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        metrics_rows = []
        episodes_total_running = 0.0
        for i in range(self.config.iterations):
            result = algo.train()
            reward_mean = self._extract_float(
                result,
                [
                    ("episode_reward_mean",),
                    ("env_runners", "episode_return_mean"),
                ],
                default=0.0,
            )
            episodes_total = self._extract_float(
                result,
                [
                    ("episodes_total",),
                    ("num_episodes_lifetime",),
                    ("sampler_results", "episodes_total"),
                    ("env_runners", "num_episodes_lifetime"),
                    ("env_runners", "num_episodes"),
                    ("env_runners", "episodes_total"),
                ],
                default=0.0,
            )
            episodes_this_iter = self._extract_float(
                result,
                [
                    ("episodes_this_iter",),
                    ("env_runners", "episodes_this_iter"),
                    ("env_runners", "num_episodes"),
                ],
                default=0.0,
            )
            if episodes_total <= 0.0:
                episodes_total_running += episodes_this_iter
                episodes_total = episodes_total_running
            else:
                episodes_total_running = episodes_total
            survival_mean = self._extract_float(
                result,
                [
                    ("episode_len_mean",),
                    ("env_runners", "episode_len_mean"),
                ],
                default=0.0,
            )
            win_rate = self._extract_float(
                result,
                [
                    ("custom_metrics", "win_rate_mean"),
                    ("env_runners", "custom_metrics", "win_rate_mean"),
                ],
                default=0.0,
            )
            starvation_rate = self._extract_float(
                result,
                [
                    ("custom_metrics", "starvation_rate_mean"),
                    ("env_runners", "custom_metrics", "starvation_rate_mean"),
                ],
                default=0.0,
            )
            team_dist = self._extract_float(
                result,
                [
                    ("custom_metrics", "mean_teammate_distance_mean"),
                    ("env_runners", "custom_metrics", "mean_teammate_distance_mean"),
                ],
                default=0.0,
            )
            loss = self._extract_loss(result)

            recent_returns.append(reward_mean)
            recent_wins.append(win_rate)
            recent_survival.append(survival_mean)
            recent_starvation.append(starvation_rate)
            recent_loss.append(loss)
            recent_teammate_dist.append(team_dist)

            metrics_rows.append(
                {
                    "iteration": i + 1,
                    "episode_reward_mean": reward_mean,
                    "episodes_total": episodes_total,
                    "timesteps_total": result.get("timesteps_total"),
                    "win_rate": win_rate,
                    "survival_mean": survival_mean,
                    "starvation_rate": starvation_rate,
                    "loss": loss,
                    "mean_teammate_distance": team_dist,
                }
            )
            self._print_live_progress(
                iteration=i + 1,
                total=self.config.iterations,
                window=window,
                ret=sum(recent_returns) / len(recent_returns),
                win=sum(recent_wins) / len(recent_wins),
                surv=sum(recent_survival) / len(recent_survival),
                starve=sum(recent_starvation) / len(recent_starvation),
                loss=sum(recent_loss) / len(recent_loss),
                team_dist=sum(recent_teammate_dist) / len(recent_teammate_dist),
                episodes_total=int(episodes_total),
            )

        print()
        checkpoint_dir = (out / "checkpoint").resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = self._save_checkpoint(algo=algo, checkpoint_dir=checkpoint_dir)
        algo.stop()

        metrics_path = out / "rllib_metrics.json"
        metrics_path.write_text(json.dumps(metrics_rows, indent=2), encoding="utf-8")

        summary = {
            "iterations": self.config.iterations,
            "checkpoint": checkpoint,
            "metrics": str(metrics_path),
        }
        summary_path = out / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return summary

    def _save_checkpoint(self, algo, checkpoint_dir: Path) -> str:
        try:
            # New API stack path.
            ckpt = algo.save_to_path(checkpoint_dir.as_posix())
            return str(ckpt)
        except RuntimeError as exc:
            if "old API stack" not in str(exc):
                raise
            # Old API stack fallback.
            ckpt = algo.save(checkpoint_dir=checkpoint_dir.as_posix())
            return str(ckpt)

    def _extract_float(self, result: Dict[str, object], paths, default: float = 0.0) -> float:
        for path in paths:
            cur = result
            ok = True
            for key in path:
                if not isinstance(cur, dict) or key not in cur:
                    ok = False
                    break
                cur = cur[key]
            if ok and isinstance(cur, numbers.Real):
                return float(cur)
        return default

    def _extract_loss(self, result: Dict[str, object]) -> float:
        info = result.get("info", {})
        if isinstance(info, dict):
            learner = info.get("learner", {})
            if isinstance(learner, dict):
                losses = []
                for _, pdata in learner.items():
                    if not isinstance(pdata, dict):
                        continue
                    stats = pdata.get("learner_stats", pdata)
                    if isinstance(stats, dict):
                        for k, v in stats.items():
                            if isinstance(v, (int, float)) and "loss" in str(k).lower():
                                losses.append(float(v))
                if losses:
                    return sum(losses) / len(losses)
        return 0.0

    def _print_live_progress(
        self,
        iteration: int,
        total: int,
        window: int,
        ret: float,
        win: float,
        surv: float,
        starve: float,
        loss: float,
        team_dist: float,
        episodes_total: int,
    ) -> None:
        bar_width = 26
        frac = iteration / max(1, total)
        filled = int(bar_width * frac)
        bar = "#" * filled + "-" * (bar_width - filled)
        line = (
            f"\r[{bar}] {iteration}/{total} "
            f"ret{window}={ret:.3f} "
            f"win{window}={win:.3f} "
            f"surv{window}={surv:.1f} "
            f"starve{window}={starve:.3f} "
            f"loss{window}={loss:.4f} "
            f"team_dist{window}={team_dist:.2f} "
            f"episodes_total={episodes_total}"
        )
        print(line, end="", flush=True)
