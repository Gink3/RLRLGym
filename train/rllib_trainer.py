"""RLlib-based trainer for RLRLGym."""

from __future__ import annotations

import json
import logging
import numbers
import os
import shutil
import warnings
from collections.abc import MutableMapping
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
from rlrlgym.curriculum import load_curriculum_phases


@dataclass
class RLlibTrainConfig:
    iterations: int = 100
    seed: int = 0
    output_dir: str = "outputs/train_rllib"
    width: Optional[int] = None
    height: Optional[int] = None
    n_agents: Optional[int] = None
    max_steps: Optional[int] = None
    framework: str = "torch"
    num_gpus: int = 0
    num_rollout_workers: int = 0
    train_batch_size: int = 4000
    replay_save_every: int = 1000
    env_config_path: str = "data/env_config.json"
    curriculum_path: str = "data/curriculum_phases.json"
    shared_policy: bool = False
    curriculum_enabled: bool = True


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
            from ray.rllib.core.rl_module.multi_rl_module import MultiRLModuleSpec
            from ray.rllib.core.rl_module.rl_module import RLModuleSpec
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
        self._MultiRLModuleSpec = MultiRLModuleSpec
        self._RLModuleSpec = RLModuleSpec
        self._register_env = register_env
        self._RLRLGymRLlibEnv = RLRLGymRLlibEnv
        self._configure_ray_storage_defaults()
        # Reduce noisy transitional warnings emitted through logging channels.
        logging.getLogger("ray.rllib.algorithms.algorithm_config").setLevel(logging.ERROR)
        logging.getLogger("ray.tune.logger.unified").setLevel(logging.ERROR)
        logging.getLogger("ray.rllib.utils.deprecation").setLevel(logging.ERROR)
        logging.getLogger("ray._common.deprecation").setLevel(logging.ERROR)
        logging.getLogger("ray.rllib.core.rl_module.rl_module").setLevel(logging.ERROR)
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
        recent_agent0_wins: deque[float] = deque(maxlen=window)
        recent_agent1_wins: deque[float] = deque(maxlen=window)
        recent_ties: deque[float] = deque(maxlen=window)
        recent_survival: deque[float] = deque(maxlen=window)
        recent_loss: deque[float] = deque(maxlen=window)

        env_config = {
            "render_enabled": False,
            "agent_profile_map": {"agent_0": "human", "agent_1": "orc"},
            "replay_save_every": int(self.config.replay_save_every),
            "replay_output_dir": str(Path(self.config.output_dir).resolve()),
            "save_latest_replay": True,
            "env_config_path": self.config.env_config_path,
            "curriculum_phases": (
                load_curriculum_phases(self.config.curriculum_path)
                if self.config.curriculum_enabled
                else []
            ),
        }
        if self.config.width is not None:
            env_config["width"] = int(self.config.width)
        if self.config.height is not None:
            env_config["height"] = int(self.config.height)
        if self.config.max_steps is not None:
            env_config["max_steps"] = int(self.config.max_steps)
        if self.config.n_agents is not None:
            env_config["n_agents"] = int(self.config.n_agents)

        self._register_env(env_name, lambda cfg: self._RLRLGymRLlibEnv(cfg))
        probe_env = self._RLRLGymRLlibEnv(env_config)
        sample_obs_space = probe_env.observation_spaces["agent_0"]
        sample_action_space = probe_env.action_spaces["agent_0"]

        class MetricsCallbacks(self._DefaultCallbacks):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self._episode_action_counts: Dict[int, Dict[str, int]] = {}

            def _get_episode_store(self, episode):
                custom = getattr(episode, "custom_data", None)
                if isinstance(custom, MutableMapping):
                    return custom
                key = id(episode)
                if key not in self._episode_action_counts:
                    self._episode_action_counts[key] = {}
                return self._episode_action_counts[key]

            def _emit_metric(self, episode, metrics_logger, key: str, value: float) -> None:
                # New RLlib API stack uses metrics_logger in callbacks.
                if metrics_logger is not None:
                    try:
                        metrics_logger.log_value(
                            ("custom_metrics", key),
                            float(value),
                            reduce="mean",
                        )
                        return
                    except Exception:
                        pass
                # Backward-compatible fallback for older Episode objects.
                cm = getattr(episode, "custom_metrics", None)
                if isinstance(cm, MutableMapping):
                    cm[key] = float(value)

            def on_episode_start(self, *, episode, **kwargs):
                store = self._get_episode_store(episode)
                store["action_counts"] = {
                    "move": 0,
                    "wait": 0,
                    "interact": 0,
                    "other": 0,
                    "total": 0,
                }

            def on_episode_step(self, *, episode, **kwargs):
                store = self._get_episode_store(episode)
                counts = store.get("action_counts")
                if not isinstance(counts, dict):
                    return
                agent_ids = []
                if hasattr(episode, "get_agents"):
                    agent_ids = list(episode.get_agents())
                elif hasattr(episode, "_agent_to_last_info"):
                    agent_ids = list(episode._agent_to_last_info.keys())
                for aid in agent_ids:
                    info = episode.last_info_for(aid)
                    if not info:
                        continue
                    action = info.get("action")
                    if action is None:
                        continue
                    try:
                        a = int(action)
                    except Exception:
                        continue
                    if a in (0, 1, 2, 3):
                        counts["move"] += 1
                    elif a == 4:
                        counts["wait"] += 1
                    elif a == 10:
                        counts["interact"] += 1
                    else:
                        counts["other"] += 1
                    counts["total"] += 1

            def on_episode_end(self, *, episode, metrics_logger=None, **kwargs):
                agent_ids = []
                if hasattr(episode, "get_agents"):
                    agent_ids = list(episode.get_agents())
                elif hasattr(episode, "_agent_to_last_info"):
                    agent_ids = list(episode._agent_to_last_info.keys())

                alive_flags = []
                starvation_flags = []
                teammate_dists = []
                winner_tag = None
                for aid in agent_ids:
                    info = episode.last_info_for(aid)
                    if not info:
                        continue
                    alive = bool(info.get("alive", False))
                    events = info.get("events", [])
                    for e in events:
                        if not isinstance(e, str) or not e.startswith("winner:"):
                            continue
                        if e == "winner:none":
                            winner_tag = "tie"
                        elif e == "winner:agent_0":
                            winner_tag = "agent_0"
                        elif e == "winner:agent_1":
                            winner_tag = "agent_1"
                    starved = bool("starve_tick" in events and "death" in events)
                    td = info.get("teammate_distance")
                    alive_flags.append(1.0 if alive else 0.0)
                    starvation_flags.append(1.0 if starved else 0.0)
                    if td is not None:
                        teammate_dists.append(float(td))

                if winner_tag is None and alive_flags:
                    # Fallback for episodes lacking explicit winner tag in events.
                    n_alive = int(sum(1 for v in alive_flags if v > 0))
                    if n_alive == 0:
                        winner_tag = "tie"
                    elif n_alive == 1:
                        # Ambiguous without explicit tag; leave as tie fallback.
                        winner_tag = "tie"
                    else:
                        winner_tag = "tie"
                self._emit_metric(
                    episode, metrics_logger, "agent0_win", 1.0 if winner_tag == "agent_0" else 0.0
                )
                self._emit_metric(
                    episode, metrics_logger, "agent1_win", 1.0 if winner_tag == "agent_1" else 0.0
                )
                self._emit_metric(
                    episode, metrics_logger, "tie", 1.0 if winner_tag == "tie" else 0.0
                )
                store = self._get_episode_store(episode)
                counts = store.get("action_counts", {})
                total_actions = max(1, int(counts.get("total", 0)))
                self._emit_metric(
                    episode,
                    metrics_logger,
                    "action_wait_rate",
                    float(counts.get("wait", 0)) / total_actions,
                )
                self._emit_metric(
                    episode,
                    metrics_logger,
                    "action_move_rate",
                    float(counts.get("move", 0)) / total_actions,
                )
                self._emit_metric(
                    episode,
                    metrics_logger,
                    "action_interact_rate",
                    float(counts.get("interact", 0)) / total_actions,
                )
                self._episode_action_counts.pop(id(episode), None)

        # Reduce warning noise from known Ray transitional deprecations.
        warnings.filterwarnings("ignore", category=DeprecationWarning, module=r"ray\..*")
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"ray\..*")

        if self.config.shared_policy:
            policy_ids = ["shared_policy"]
            rl_module_spec = self._MultiRLModuleSpec(
                rl_module_specs={
                    "shared_policy": self._RLModuleSpec(
                        observation_space=sample_obs_space,
                        action_space=sample_action_space,
                        inference_only=False,
                        model_config={},
                    )
                }
            )

            def policy_mapping_fn(agent_id, *args, **kwargs):
                return "shared_policy"
        else:
            policy_ids = ["human_policy", "orc_policy"]
            rl_module_spec = self._MultiRLModuleSpec(
                rl_module_specs={
                    "human_policy": self._RLModuleSpec(
                        observation_space=sample_obs_space,
                        action_space=sample_action_space,
                        inference_only=False,
                        model_config={},
                    ),
                    "orc_policy": self._RLModuleSpec(
                        observation_space=sample_obs_space,
                        action_space=sample_action_space,
                        inference_only=False,
                        model_config={},
                    ),
                }
            )

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
            .rl_module(rl_module_spec=rl_module_spec)
            .callbacks(MetricsCallbacks)
            .multi_agent(
                policies=policy_ids,
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
            agent0_win = self._extract_float(
                result,
                [
                    ("custom_metrics", "agent0_win_mean"),
                    ("custom_metrics", "agent0_win"),
                    ("env_runners", "custom_metrics", "agent0_win_mean"),
                    ("env_runners", "custom_metrics", "agent0_win"),
                ],
                default=0.0,
            )
            agent1_win = self._extract_float(
                result,
                [
                    ("custom_metrics", "agent1_win_mean"),
                    ("custom_metrics", "agent1_win"),
                    ("env_runners", "custom_metrics", "agent1_win_mean"),
                    ("env_runners", "custom_metrics", "agent1_win"),
                ],
                default=0.0,
            )
            tie_rate = self._extract_float(
                result,
                [
                    ("custom_metrics", "tie_mean"),
                    ("custom_metrics", "tie"),
                    ("env_runners", "custom_metrics", "tie_mean"),
                    ("env_runners", "custom_metrics", "tie"),
                ],
                default=0.0,
            )
            action_wait_rate = self._extract_float(
                result,
                [
                    ("custom_metrics", "action_wait_rate_mean"),
                    ("custom_metrics", "action_wait_rate"),
                    ("env_runners", "custom_metrics", "action_wait_rate_mean"),
                    ("env_runners", "custom_metrics", "action_wait_rate"),
                ],
                default=0.0,
            )
            action_move_rate = self._extract_float(
                result,
                [
                    ("custom_metrics", "action_move_rate_mean"),
                    ("custom_metrics", "action_move_rate"),
                    ("env_runners", "custom_metrics", "action_move_rate_mean"),
                    ("env_runners", "custom_metrics", "action_move_rate"),
                ],
                default=0.0,
            )
            action_interact_rate = self._extract_float(
                result,
                [
                    ("custom_metrics", "action_interact_rate_mean"),
                    ("custom_metrics", "action_interact_rate"),
                    ("env_runners", "custom_metrics", "action_interact_rate_mean"),
                    ("env_runners", "custom_metrics", "action_interact_rate"),
                ],
                default=0.0,
            )
            loss = self._extract_loss(result)

            recent_returns.append(reward_mean)
            recent_agent0_wins.append(agent0_win)
            recent_agent1_wins.append(agent1_win)
            recent_ties.append(tie_rate)
            recent_survival.append(survival_mean)
            recent_loss.append(loss)

            metrics_rows.append(
                {
                    "iteration": i + 1,
                    "episode_reward_mean": reward_mean,
                    "episodes_total": episodes_total,
                    "timesteps_total": result.get("timesteps_total"),
                    "win_rate": agent0_win + agent1_win,
                    "survival_mean": survival_mean,
                    "starvation_rate": 0.0,
                    "loss": loss,
                    "mean_teammate_distance": 0.0,
                    "agent0_win": agent0_win,
                    "agent1_win": agent1_win,
                    "tie": tie_rate,
                    "action_wait_rate": action_wait_rate,
                    "action_move_rate": action_move_rate,
                    "action_interact_rate": action_interact_rate,
                }
            )
            self._print_live_progress(
                iteration=i + 1,
                total=self.config.iterations,
                window=window,
                ret=sum(recent_returns) / len(recent_returns),
                agent0_win_count=int(round(sum(recent_agent0_wins))),
                agent1_win_count=int(round(sum(recent_agent1_wins))),
                tie_count=int(round(sum(recent_ties))),
                surv=sum(recent_survival) / len(recent_survival),
                loss=sum(recent_loss) / len(recent_loss),
                episodes_total=int(episodes_total),
            )

        print()
        checkpoint_dir = (out / "checkpoint").resolve()
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = self._save_checkpoint(algo=algo, checkpoint_dir=checkpoint_dir)
        algo.stop()

        metrics_path = out / "rllib_metrics.json"
        metrics_path.write_text(json.dumps(metrics_rows, indent=2), encoding="utf-8")
        episodes_total_final = int(metrics_rows[-1]["episodes_total"]) if metrics_rows else 0
        self._ensure_final_replay_exists(
            output_dir=out, final_episode=episodes_total_final
        )

        summary = {
            "iterations": self.config.iterations,
            "episodes_total": episodes_total_final,
            "checkpoint": checkpoint,
            "metrics": str(metrics_path),
            "replay_dir": str((out / "replays").resolve()),
            "replay_save_every": int(self.config.replay_save_every),
        }
        dashboard_path = out / "dashboard.html"
        dashboard_path.write_text(
            self._build_dashboard_html(metrics_rows=metrics_rows, summary=summary),
            encoding="utf-8",
        )
        summary["dashboard"] = str(dashboard_path)
        summary_path = out / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return summary

    def _ensure_final_replay_exists(self, output_dir: Path, final_episode: int) -> None:
        if final_episode <= 0:
            return
        replay_dir = output_dir / "replays"
        latest = replay_dir / "latest_episode.replay.json"
        final = replay_dir / f"episode_{final_episode:06d}.replay.json"
        if final.exists() or not latest.exists():
            return
        replay_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(latest, final)

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
        agent0_win_count: int,
        agent1_win_count: int,
        tie_count: int,
        surv: float,
        loss: float,
        episodes_total: int,
    ) -> None:
        bar_width = 26
        frac = iteration / max(1, total)
        filled = int(bar_width * frac)
        bar = "#" * filled + "-" * (bar_width - filled)
        line = (
            f"\r[{bar}] {iteration}/{total} "
            f"ret{window}={ret:.3f} "
            f"a0_wins{window}={agent0_win_count} "
            f"a1_wins{window}={agent1_win_count} "
            f"ties{window}={tie_count} "
            f"surv{window}={surv:.1f} "
            f"loss{window}={loss:.4f} "
            f"episodes_total={episodes_total}"
        )
        print(line, end="", flush=True)

    def _build_dashboard_html(self, metrics_rows: list[dict], summary: dict) -> str:
        episodes_total = int(summary.get("episodes_total", 0))
        iterations = int(summary.get("iterations", 0))
        latest = metrics_rows[-1] if metrics_rows else {}
        latest_reward = float(latest.get("episode_reward_mean", 0.0) or 0.0)
        latest_win = float(latest.get("win_rate", 0.0) or 0.0)
        latest_survival = float(latest.get("survival_mean", 0.0) or 0.0)
        latest_starve = float(latest.get("starvation_rate", 0.0) or 0.0)
        latest_loss = float(latest.get("loss", 0.0) or 0.0)
        latest_team_dist = float(latest.get("mean_teammate_distance", 0.0) or 0.0)
        latest_wait_rate = float(latest.get("action_wait_rate", 0.0) or 0.0)
        latest_move_rate = float(latest.get("action_move_rate", 0.0) or 0.0)
        latest_interact_rate = float(latest.get("action_interact_rate", 0.0) or 0.0)
        latest_timesteps_total = int(float(latest.get("timesteps_total", 0) or 0))
        reward_curve = [round(float(r.get("episode_reward_mean", 0.0) or 0.0), 4) for r in metrics_rows]
        win_curve = [round(float(r.get("win_rate", 0.0) or 0.0), 4) for r in metrics_rows]
        survival_curve = [round(float(r.get("survival_mean", 0.0) or 0.0), 4) for r in metrics_rows]
        starvation_curve = [round(float(r.get("starvation_rate", 0.0) or 0.0), 4) for r in metrics_rows]
        loss_curve = [round(float(r.get("loss", 0.0) or 0.0), 6) for r in metrics_rows]
        team_dist_curve = [round(float(r.get("mean_teammate_distance", 0.0) or 0.0), 4) for r in metrics_rows]
        wait_curve = [round(float(r.get("action_wait_rate", 0.0) or 0.0), 4) for r in metrics_rows]
        move_curve = [round(float(r.get("action_move_rate", 0.0) or 0.0), 4) for r in metrics_rows]
        interact_curve = [round(float(r.get("action_interact_rate", 0.0) or 0.0), 4) for r in metrics_rows]

        rows_html = "".join(
            [
                "<tr>"
                f"<td>{int(r.get('iteration', 0))}</td>"
                f"<td>{int(float(r.get('episodes_total', 0) or 0))}</td>"
                f"<td>{int(float(r.get('timesteps_total', 0) or 0))}</td>"
                f"<td>{float(r.get('episode_reward_mean', 0.0) or 0.0):.4f}</td>"
                f"<td>{float(r.get('win_rate', 0.0) or 0.0):.4f}</td>"
                f"<td>{float(r.get('survival_mean', 0.0) or 0.0):.2f}</td>"
                f"<td>{float(r.get('starvation_rate', 0.0) or 0.0):.4f}</td>"
                f"<td>{float(r.get('loss', 0.0) or 0.0):.6f}</td>"
                f"<td>{float(r.get('mean_teammate_distance', 0.0) or 0.0):.4f}</td>"
                f"<td>{float(r.get('action_wait_rate', 0.0) or 0.0):.4f}</td>"
                f"<td>{float(r.get('action_move_rate', 0.0) or 0.0):.4f}</td>"
                f"<td>{float(r.get('action_interact_rate', 0.0) or 0.0):.4f}</td>"
                "</tr>"
                for r in metrics_rows
            ]
        )

        return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>RLRLGym RLlib Dashboard</title>
  <style>
    :root {{
      --bg: #12161d;
      --surface: #1d2430;
      --border: #2f3a4d;
      --text: #f2f7ff;
      --muted: #9eb0cb;
      --accent: #41c79d;
    }}
    body {{
      font-family: -apple-system, Segoe UI, sans-serif;
      margin: 24px;
      background: var(--bg);
      color: var(--text);
    }}
    .card {{
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
      margin-bottom: 16px;
    }}
    .metric {{
      display: inline-block;
      margin-right: 16px;
      margin-bottom: 6px;
      font-weight: 700;
      color: var(--accent);
    }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{
      border: 1px solid var(--border);
      padding: 6px 8px;
      text-align: left;
      font-size: 13px;
    }}
    th {{ background: #273247; }}
    pre {{
      margin: 0;
      padding: 10px;
      border-radius: 8px;
      background: #10161f;
      border: 1px solid var(--border);
      color: #d4def0;
      overflow-x: auto;
    }}
  </style>
</head>
<body>
  <h1>RLRLGym RLlib Dashboard</h1>
  <div class="card">
    <div class="metric">Episodes (from episodes_total): {episodes_total}</div>
    <div class="metric">Iterations: {iterations}</div>
    <div class="metric">Latest timesteps total: {latest_timesteps_total}</div>
    <div class="metric">Latest reward: {latest_reward:.4f}</div>
    <div class="metric">Latest win rate: {latest_win:.4f}</div>
    <div class="metric">Latest survival mean: {latest_survival:.2f}</div>
    <div class="metric">Latest starvation rate: {latest_starve:.4f}</div>
    <div class="metric">Latest loss: {latest_loss:.6f}</div>
    <div class="metric">Latest teammate dist: {latest_team_dist:.4f}</div>
    <div class="metric">Latest wait rate: {latest_wait_rate:.4f}</div>
    <div class="metric">Latest move rate: {latest_move_rate:.4f}</div>
    <div class="metric">Latest interact rate: {latest_interact_rate:.4f}</div>
  </div>
  <div class="card">
    <h3>Reward Curve (per iteration)</h3>
    <pre>{reward_curve}</pre>
  </div>
  <div class="card">
    <h3>All Metric Curves (per iteration)</h3>
    <pre>win_rate={win_curve}</pre>
    <pre>survival_mean={survival_curve}</pre>
    <pre>starvation_rate={starvation_curve}</pre>
    <pre>loss={loss_curve}</pre>
    <pre>mean_teammate_distance={team_dist_curve}</pre>
    <pre>action_wait_rate={wait_curve}</pre>
    <pre>action_move_rate={move_curve}</pre>
    <pre>action_interact_rate={interact_curve}</pre>
  </div>
  <div class="card">
    <h3>Iteration Metrics</h3>
    <table>
      <thead>
        <tr><th>Iteration</th><th>Episodes Total</th><th>Timesteps Total</th><th>Reward Mean</th><th>Win Rate</th><th>Survival Mean</th><th>Starvation Rate</th><th>Loss</th><th>Team Dist</th><th>Wait</th><th>Move</th><th>Interact</th></tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>
</body>
</html>"""
