"""RLlib-based trainer for RLRLGym."""

from __future__ import annotations

import json
import logging
import numbers
import os
import shutil
import subprocess
import sys
import warnings
from collections.abc import MutableMapping
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple
from rlrlgym.curriculum import load_curriculum_phases

from .aim_logger import AimLogger

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
    replay_save_every: int = 5000
    env_config_path: str = "data/env_config.json"
    scenario_path: str = ""
    curriculum_path: str = "data/base/curriculum_phases.json"
    shared_policy: bool = False
    curriculum_enabled: bool = True
    aim_enabled: bool = True
    aim_experiment: str = "rlrlgym_rllib"
    aim_repo_path: str = "/proj/aimml"


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
        self.aim = AimLogger(
            enabled=config.aim_enabled,
            experiment=config.aim_experiment,
            repo_path=config.aim_repo_path,
            run_name="rllib_trainer",
        )
        self.aim.set_params(
            {
                "backend": "rllib",
                "iterations": int(config.iterations),
                "seed": int(config.seed),
                "output_dir": str(config.output_dir),
                "width": None if config.width is None else int(config.width),
                "height": None if config.height is None else int(config.height),
                "n_agents": None if config.n_agents is None else int(config.n_agents),
                "max_steps": None if config.max_steps is None else int(config.max_steps),
                "framework": str(config.framework),
                "num_gpus": float(config.num_gpus),
                "num_rollout_workers": int(config.num_rollout_workers),
                "train_batch_size": int(config.train_batch_size),
                "replay_save_every": int(config.replay_save_every),
                "env_config_path": str(config.env_config_path),
                "scenario_path": str(config.scenario_path or ""),
                "curriculum_path": str(config.curriculum_path),
                "shared_policy": bool(config.shared_policy),
                "curriculum_enabled": bool(config.curriculum_enabled),
                "aim_repo_path": str(config.aim_repo_path),
            }
        )
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

        curriculum_phases = (
            load_curriculum_phases(self.config.curriculum_path)
            if self.config.curriculum_enabled
            else []
        )
        monster_name_map = self._load_monster_name_map(self.config.env_config_path)
        env_config = {
            "render_enabled": False,
            "replay_save_every": int(self.config.replay_save_every),
            "replay_output_dir": str(Path(self.config.output_dir).resolve()),
            "save_latest_replay": True,
            "env_config_path": self.config.env_config_path,
            "scenario_path": str(self.config.scenario_path or ""),
            "curriculum_phases": curriculum_phases,
        }
        if self.config.width is not None:
            env_config["width"] = int(self.config.width)
        if self.config.height is not None:
            env_config["height"] = int(self.config.height)
        if self.config.max_steps is not None:
            env_config["max_steps"] = int(self.config.max_steps)
        if self.config.n_agents is not None:
            env_config["n_agents"] = int(self.config.n_agents)

        env_cls = self._RLRLGymRLlibEnv
        self._register_env(
            env_name,
            lambda cfg, _env_cls=env_cls: _env_cls(cfg),
        )
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
                store["reward_components"] = {
                    "action_total": 0.0,
                    "survival": 0.0,
                    "search_explore": 0.0,
                    "profile_shape": 0.0,
                    "terminal": 0.0,
                }
                store["damage_events"] = 0
                store["kill_events"] = 0
                store["phase_index"] = 0.0

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
                    rc = info.get("reward_components")
                    if isinstance(rc, dict):
                        rcs = store.get("reward_components")
                        if isinstance(rcs, dict):
                            for k in ("action_total", "survival", "search_explore", "profile_shape", "terminal"):
                                val = rc.get(k)
                                if isinstance(val, numbers.Real):
                                    rcs[k] = float(rcs.get(k, 0.0)) + float(val)
                    events = info.get("events", [])
                    if isinstance(events, list):
                        for evt in events:
                            if not isinstance(evt, str):
                                continue
                            if (
                                evt.startswith("agent_interact:hit:")
                                or evt.startswith("agent_interact:hit_monster:")
                                or evt.startswith("monster_hit:")
                            ):
                                store["damage_events"] = int(store.get("damage_events", 0)) + 1
                            if (
                                evt.startswith("agent_interact:kill:")
                                or evt.startswith("agent_interact:kill_monster:")
                                or evt.startswith("death_by_monster:")
                            ):
                                store["kill_events"] = int(store.get("kill_events", 0)) + 1
                    pidx = info.get("phase_index")
                    if isinstance(pidx, numbers.Real):
                        store["phase_index"] = max(float(store.get("phase_index", 0.0)), float(pidx))

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
                death_counts = {
                    "starvation": 0,
                    "monster": 0,
                    "agent": 0,
                    "other": 0,
                }
                death_by_monster: Dict[str, int] = {}
                coverages = []
                stagnation_steps = []
                enemy_visible_steps = []
                enemy_distances = []
                enemy_distance_delta_means = []
                first_seen_steps = []
                first_seen_count = 0
                combat_exchange_counts = []
                timeout_no_contact_flags = []
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
                    cov = info.get("explore_coverage")
                    if cov is not None:
                        coverages.append(float(cov))
                    ssn = info.get("steps_since_new_tile")
                    if ssn is not None:
                        stagnation_steps.append(float(ssn))
                    evs = info.get("enemy_visible_steps")
                    if evs is not None:
                        enemy_visible_steps.append(float(evs))
                    ed = info.get("enemy_distance")
                    if ed is not None:
                        enemy_distances.append(float(ed))
                    edd = info.get("enemy_distance_delta_mean")
                    if edd is not None:
                        enemy_distance_delta_means.append(float(edd))
                    fes = info.get("first_enemy_seen_step")
                    if fes is not None:
                        first_seen_count += 1
                        first_seen_steps.append(float(fes))
                    cex = info.get("combat_exchanges")
                    if cex is not None:
                        combat_exchange_counts.append(float(cex))
                    tnc = info.get("timeout_no_contact")
                    if tnc is not None:
                        timeout_no_contact_flags.append(1.0 if bool(tnc) else 0.0)
                    if not alive:
                        death_reason = str(info.get("death_reason", ""))
                        if starved:
                            death_counts["starvation"] += 1
                        elif any(
                            isinstance(evt, str) and evt.startswith("death_by_monster:")
                            for evt in events
                        ):
                            death_counts["monster"] += 1
                            for evt in events:
                                if isinstance(evt, str) and evt.startswith("death_by_monster:"):
                                    monster_id = evt.split(":", 1)[1].strip()
                                    if monster_id:
                                        death_by_monster[monster_id] = (
                                            death_by_monster.get(monster_id, 0) + 1
                                        )
                                    break
                        elif death_reason.startswith("killed by agent"):
                            death_counts["agent"] += 1
                        else:
                            death_counts["other"] += 1

                if winner_tag is None:
                    # If no explicit winner event was emitted, treat as tie.
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
                n_agents = max(1, len(agent_ids))
                self._emit_metric(
                    episode,
                    metrics_logger,
                    "explore_coverage",
                    (sum(coverages) / len(coverages)) if coverages else 0.0,
                )
                self._emit_metric(
                    episode,
                    metrics_logger,
                    "steps_since_new_tile",
                    (sum(stagnation_steps) / len(stagnation_steps))
                    if stagnation_steps
                    else 0.0,
                )
                self._emit_metric(
                    episode,
                    metrics_logger,
                    "enemy_visible_steps",
                    (sum(enemy_visible_steps) / len(enemy_visible_steps))
                    if enemy_visible_steps
                    else 0.0,
                )
                self._emit_metric(
                    episode,
                    metrics_logger,
                    "enemy_distance",
                    (sum(enemy_distances) / len(enemy_distances))
                    if enemy_distances
                    else 0.0,
                )
                self._emit_metric(
                    episode,
                    metrics_logger,
                    "enemy_distance_delta_mean",
                    (sum(enemy_distance_delta_means) / len(enemy_distance_delta_means))
                    if enemy_distance_delta_means
                    else 0.0,
                )
                self._emit_metric(
                    episode,
                    metrics_logger,
                    "first_enemy_seen_rate",
                    float(first_seen_count) / float(n_agents),
                )
                self._emit_metric(
                    episode,
                    metrics_logger,
                    "first_enemy_seen_step",
                    (sum(first_seen_steps) / len(first_seen_steps))
                    if first_seen_steps
                    else 0.0,
                )
                self._emit_metric(
                    episode,
                    metrics_logger,
                    "combat_exchanges",
                    (sum(combat_exchange_counts) / len(combat_exchange_counts))
                    if combat_exchange_counts
                    else 0.0,
                )
                self._emit_metric(
                    episode,
                    metrics_logger,
                    "timeout_no_contact_rate",
                    (sum(timeout_no_contact_flags) / len(timeout_no_contact_flags))
                    if timeout_no_contact_flags
                    else 0.0,
                )
                self._emit_metric(
                    episode,
                    metrics_logger,
                    "engagement_rate",
                    1.0 if any(v > 0.0 for v in combat_exchange_counts) else 0.0,
                )
                self._emit_metric(
                    episode,
                    metrics_logger,
                    "damage_event_count",
                    float(store.get("damage_events", 0)),
                )
                self._emit_metric(
                    episode,
                    metrics_logger,
                    "kill_event_count",
                    float(store.get("kill_events", 0)),
                )
                self._emit_metric(
                    episode,
                    metrics_logger,
                    "phase_index",
                    float(store.get("phase_index", 0.0)),
                )
                rcs = store.get("reward_components", {})
                if isinstance(rcs, dict):
                    self._emit_metric(
                        episode,
                        metrics_logger,
                        "reward_comp_action_total",
                        float(rcs.get("action_total", 0.0)),
                    )
                    self._emit_metric(
                        episode,
                        metrics_logger,
                        "reward_comp_survival",
                        float(rcs.get("survival", 0.0)),
                    )
                    self._emit_metric(
                        episode,
                        metrics_logger,
                        "reward_comp_search_explore",
                        float(rcs.get("search_explore", 0.0)),
                    )
                    self._emit_metric(
                        episode,
                        metrics_logger,
                        "reward_comp_profile_shape",
                        float(rcs.get("profile_shape", 0.0)),
                    )
                    self._emit_metric(
                        episode,
                        metrics_logger,
                        "reward_comp_terminal",
                        float(rcs.get("terminal", 0.0)),
                    )
                self._emit_metric(
                    episode,
                    metrics_logger,
                    "death_starvation",
                    float(death_counts["starvation"]) / n_agents,
                )
                self._emit_metric(
                    episode,
                    metrics_logger,
                    "death_monster",
                    float(death_counts["monster"]) / n_agents,
                )
                self._emit_metric(
                    episode,
                    metrics_logger,
                    "death_agent",
                    float(death_counts["agent"]) / n_agents,
                )
                self._emit_metric(
                    episode,
                    metrics_logger,
                    "death_other",
                    float(death_counts["other"]) / n_agents,
                )
                for monster_id, cnt in death_by_monster.items():
                    self._emit_metric(
                        episode,
                        metrics_logger,
                        f"death_by_monster__{monster_id}",
                        float(cnt) / n_agents,
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
        death_histogram = {
            "starvation": 0,
            "monster": 0,
            "agent": 0,
            "other": 0,
        }
        death_by_monster_histogram: Dict[str, int] = {}
        n_agents = int(env_config.get("n_agents", 2))
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
            tie_metric_raw = self._extract_float(
                result,
                [
                    ("custom_metrics", "tie_mean"),
                    ("custom_metrics", "tie"),
                    ("env_runners", "custom_metrics", "tie_mean"),
                    ("env_runners", "custom_metrics", "tie"),
                ],
                default=0.0,
            )
            # For two-agent episodes with mutually exclusive outcomes, derive tie
            # directly from wins to avoid callback metric key drift.
            tie_rate = max(0.0, min(1.0, 1.0 - agent0_win - agent1_win))
            if tie_metric_raw > 0.0:
                tie_rate = float(tie_metric_raw)
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
            explore_coverage = self._extract_float(
                result,
                [
                    ("custom_metrics", "explore_coverage_mean"),
                    ("custom_metrics", "explore_coverage"),
                    ("env_runners", "custom_metrics", "explore_coverage_mean"),
                    ("env_runners", "custom_metrics", "explore_coverage"),
                ],
                default=0.0,
            )
            steps_since_new_tile = self._extract_float(
                result,
                [
                    ("custom_metrics", "steps_since_new_tile_mean"),
                    ("custom_metrics", "steps_since_new_tile"),
                    ("env_runners", "custom_metrics", "steps_since_new_tile_mean"),
                    ("env_runners", "custom_metrics", "steps_since_new_tile"),
                ],
                default=0.0,
            )
            enemy_visible_steps = self._extract_float(
                result,
                [
                    ("custom_metrics", "enemy_visible_steps_mean"),
                    ("custom_metrics", "enemy_visible_steps"),
                    ("env_runners", "custom_metrics", "enemy_visible_steps_mean"),
                    ("env_runners", "custom_metrics", "enemy_visible_steps"),
                ],
                default=0.0,
            )
            enemy_distance = self._extract_float(
                result,
                [
                    ("custom_metrics", "enemy_distance_mean"),
                    ("custom_metrics", "enemy_distance"),
                    ("env_runners", "custom_metrics", "enemy_distance_mean"),
                    ("env_runners", "custom_metrics", "enemy_distance"),
                ],
                default=0.0,
            )
            enemy_distance_delta_mean = self._extract_float(
                result,
                [
                    ("custom_metrics", "enemy_distance_delta_mean_mean"),
                    ("custom_metrics", "enemy_distance_delta_mean"),
                    ("env_runners", "custom_metrics", "enemy_distance_delta_mean_mean"),
                    ("env_runners", "custom_metrics", "enemy_distance_delta_mean"),
                ],
                default=0.0,
            )
            first_enemy_seen_rate = self._extract_float(
                result,
                [
                    ("custom_metrics", "first_enemy_seen_rate_mean"),
                    ("custom_metrics", "first_enemy_seen_rate"),
                    ("env_runners", "custom_metrics", "first_enemy_seen_rate_mean"),
                    ("env_runners", "custom_metrics", "first_enemy_seen_rate"),
                ],
                default=0.0,
            )
            first_enemy_seen_step = self._extract_float(
                result,
                [
                    ("custom_metrics", "first_enemy_seen_step_mean"),
                    ("custom_metrics", "first_enemy_seen_step"),
                    ("env_runners", "custom_metrics", "first_enemy_seen_step_mean"),
                    ("env_runners", "custom_metrics", "first_enemy_seen_step"),
                ],
                default=0.0,
            )
            combat_exchanges = self._extract_float(
                result,
                [
                    ("custom_metrics", "combat_exchanges_mean"),
                    ("custom_metrics", "combat_exchanges"),
                    ("env_runners", "custom_metrics", "combat_exchanges_mean"),
                    ("env_runners", "custom_metrics", "combat_exchanges"),
                ],
                default=0.0,
            )
            timeout_no_contact_rate = self._extract_float(
                result,
                [
                    ("custom_metrics", "timeout_no_contact_rate_mean"),
                    ("custom_metrics", "timeout_no_contact_rate"),
                    ("env_runners", "custom_metrics", "timeout_no_contact_rate_mean"),
                    ("env_runners", "custom_metrics", "timeout_no_contact_rate"),
                ],
                default=0.0,
            )
            engagement_rate = self._extract_float(
                result,
                [
                    ("custom_metrics", "engagement_rate_mean"),
                    ("custom_metrics", "engagement_rate"),
                    ("env_runners", "custom_metrics", "engagement_rate_mean"),
                    ("env_runners", "custom_metrics", "engagement_rate"),
                ],
                default=0.0,
            )
            damage_event_count = self._extract_float(
                result,
                [
                    ("custom_metrics", "damage_event_count_mean"),
                    ("custom_metrics", "damage_event_count"),
                    ("env_runners", "custom_metrics", "damage_event_count_mean"),
                    ("env_runners", "custom_metrics", "damage_event_count"),
                ],
                default=0.0,
            )
            kill_event_count = self._extract_float(
                result,
                [
                    ("custom_metrics", "kill_event_count_mean"),
                    ("custom_metrics", "kill_event_count"),
                    ("env_runners", "custom_metrics", "kill_event_count_mean"),
                    ("env_runners", "custom_metrics", "kill_event_count"),
                ],
                default=0.0,
            )
            phase_index = self._extract_float(
                result,
                [
                    ("custom_metrics", "phase_index_mean"),
                    ("custom_metrics", "phase_index"),
                    ("env_runners", "custom_metrics", "phase_index_mean"),
                    ("env_runners", "custom_metrics", "phase_index"),
                ],
                default=0.0,
            )
            if phase_index <= 0.0:
                phase_index = float(
                    self._phase_index_for_episode(
                        episode=int(episodes_total),
                        phases=curriculum_phases,
                    )
                )
            phase_name = self._phase_name_for_index(
                phase_index=int(round(phase_index)),
                phases=curriculum_phases,
            )
            reward_comp_action_total = self._extract_float(
                result,
                [
                    ("custom_metrics", "reward_comp_action_total_mean"),
                    ("custom_metrics", "reward_comp_action_total"),
                    ("env_runners", "custom_metrics", "reward_comp_action_total_mean"),
                    ("env_runners", "custom_metrics", "reward_comp_action_total"),
                ],
                default=0.0,
            )
            reward_comp_survival = self._extract_float(
                result,
                [
                    ("custom_metrics", "reward_comp_survival_mean"),
                    ("custom_metrics", "reward_comp_survival"),
                    ("env_runners", "custom_metrics", "reward_comp_survival_mean"),
                    ("env_runners", "custom_metrics", "reward_comp_survival"),
                ],
                default=0.0,
            )
            reward_comp_search_explore = self._extract_float(
                result,
                [
                    ("custom_metrics", "reward_comp_search_explore_mean"),
                    ("custom_metrics", "reward_comp_search_explore"),
                    ("env_runners", "custom_metrics", "reward_comp_search_explore_mean"),
                    ("env_runners", "custom_metrics", "reward_comp_search_explore"),
                ],
                default=0.0,
            )
            reward_comp_profile_shape = self._extract_float(
                result,
                [
                    ("custom_metrics", "reward_comp_profile_shape_mean"),
                    ("custom_metrics", "reward_comp_profile_shape"),
                    ("env_runners", "custom_metrics", "reward_comp_profile_shape_mean"),
                    ("env_runners", "custom_metrics", "reward_comp_profile_shape"),
                ],
                default=0.0,
            )
            reward_comp_terminal = self._extract_float(
                result,
                [
                    ("custom_metrics", "reward_comp_terminal_mean"),
                    ("custom_metrics", "reward_comp_terminal"),
                    ("env_runners", "custom_metrics", "reward_comp_terminal_mean"),
                    ("env_runners", "custom_metrics", "reward_comp_terminal"),
                ],
                default=0.0,
            )
            policy_entropy = self._extract_metric_by_substring(
                result=result,
                substrings=("entropy",),
                default=0.0,
            )
            policy_kl = self._extract_metric_by_substring(
                result=result,
                substrings=("kl",),
                default=0.0,
            )
            death_starvation_rate = self._extract_float(
                result,
                [
                    ("custom_metrics", "death_starvation_mean"),
                    ("custom_metrics", "death_starvation"),
                    ("env_runners", "custom_metrics", "death_starvation_mean"),
                    ("env_runners", "custom_metrics", "death_starvation"),
                ],
                default=0.0,
            )
            death_monster_rate = self._extract_float(
                result,
                [
                    ("custom_metrics", "death_monster_mean"),
                    ("custom_metrics", "death_monster"),
                    ("env_runners", "custom_metrics", "death_monster_mean"),
                    ("env_runners", "custom_metrics", "death_monster"),
                ],
                default=0.0,
            )
            death_agent_rate = self._extract_float(
                result,
                [
                    ("custom_metrics", "death_agent_mean"),
                    ("custom_metrics", "death_agent"),
                    ("env_runners", "custom_metrics", "death_agent_mean"),
                    ("env_runners", "custom_metrics", "death_agent"),
                ],
                default=0.0,
            )
            death_other_rate = self._extract_float(
                result,
                [
                    ("custom_metrics", "death_other_mean"),
                    ("custom_metrics", "death_other"),
                    ("env_runners", "custom_metrics", "death_other_mean"),
                    ("env_runners", "custom_metrics", "death_other"),
                ],
                default=0.0,
            )
            custom_metrics = self._extract_custom_metric_map(result)
            loss = self._extract_loss(result)
            if episodes_this_iter > 0:
                deaths_scale = int(round(episodes_this_iter)) * max(1, n_agents)
                death_histogram["starvation"] += int(round(death_starvation_rate * deaths_scale))
                death_histogram["monster"] += int(round(death_monster_rate * deaths_scale))
                death_histogram["agent"] += int(round(death_agent_rate * deaths_scale))
                death_histogram["other"] += int(round(death_other_rate * deaths_scale))
                for key, value in custom_metrics.items():
                    if not key.startswith("death_by_monster__"):
                        continue
                    monster_id = key.split("__", 1)[1].strip()
                    if not monster_id:
                        continue
                    death_by_monster_histogram[monster_id] = (
                        death_by_monster_histogram.get(monster_id, 0)
                        + int(round(float(value) * deaths_scale))
                    )

            recent_returns.append(reward_mean)
            recent_agent0_wins.append(agent0_win)
            recent_agent1_wins.append(agent1_win)
            recent_ties.append(tie_rate)
            recent_survival.append(survival_mean)
            recent_loss.append(loss)

            row = {
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
                "explore_coverage": explore_coverage,
                "steps_since_new_tile": steps_since_new_tile,
                "enemy_visible_steps": enemy_visible_steps,
                "enemy_distance": enemy_distance,
                "enemy_distance_delta_mean": enemy_distance_delta_mean,
                "first_enemy_seen_rate": first_enemy_seen_rate,
                "first_enemy_seen_step": first_enemy_seen_step,
                "combat_exchanges": combat_exchanges,
                "timeout_no_contact_rate": timeout_no_contact_rate,
                "engagement_rate": engagement_rate,
                "damage_event_count": damage_event_count,
                "kill_event_count": kill_event_count,
                "phase_index": int(round(phase_index)),
                "phase_name": phase_name,
                "reward_comp_action_total": reward_comp_action_total,
                "reward_comp_survival": reward_comp_survival,
                "reward_comp_search_explore": reward_comp_search_explore,
                "reward_comp_profile_shape": reward_comp_profile_shape,
                "reward_comp_terminal": reward_comp_terminal,
                "policy_entropy": policy_entropy,
                "policy_kl": policy_kl,
                "death_starvation": int(death_histogram["starvation"]),
                "death_monster": int(death_histogram["monster"]),
                "death_agent": int(death_histogram["agent"]),
                "death_other": int(death_histogram["other"]),
            }
            metrics_rows.append(row)
            self.aim.track_many(
                row,
                step=i + 1,
                prefix="rllib/iteration",
            )
            self.aim.track_many(
                death_by_monster_histogram,
                step=i + 1,
                prefix="rllib/death_by_monster_histogram",
            )
            monster_iter_metrics = {
                key.split("__", 1)[1]: value
                for key, value in custom_metrics.items()
                if key.startswith("death_by_monster__")
            }
            self.aim.track_many(
                monster_iter_metrics,
                step=i + 1,
                prefix="rllib/death_by_monster_rate",
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
            "cause_of_death_histogram": self._compose_cause_histogram(
                death_histogram=death_histogram,
                death_by_monster_histogram=death_by_monster_histogram,
                monster_name_map=monster_name_map,
            ),
        }
        summary["plot_images"] = self._write_metric_plots(out=out, metrics_rows=metrics_rows)
        summary_path = out / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.aim.set_payload("rllib/summary", summary)
        self.aim.close()

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

    def _extract_custom_metric_map(self, result: Dict[str, object]) -> Dict[str, float]:
        merged: Dict[str, Tuple[int, float]] = {}

        def _ingest(blob: object) -> None:
            if not isinstance(blob, dict):
                return
            for raw_key, raw_val in blob.items():
                if not isinstance(raw_key, str):
                    continue
                if not isinstance(raw_val, numbers.Real):
                    continue
                key = raw_key
                priority = 1
                if key.endswith("_mean"):
                    key = key[: -len("_mean")]
                    priority = 2
                old = merged.get(key)
                if old is None or priority >= old[0]:
                    merged[key] = (priority, float(raw_val))

        _ingest(result.get("custom_metrics"))
        env_runners = result.get("env_runners")
        if isinstance(env_runners, dict):
            _ingest(env_runners.get("custom_metrics"))
        return {k: v for k, (_, v) in merged.items()}

    def _extract_metric_by_substring(
        self,
        result: Dict[str, object],
        substrings: Tuple[str, ...],
        default: float = 0.0,
    ) -> float:
        info = result.get("info", {})
        if not isinstance(info, dict):
            return default
        learner = info.get("learner", {})
        if not isinstance(learner, dict):
            return default
        vals: list[float] = []
        needles = tuple(s.lower() for s in substrings)
        for pdata in learner.values():
            if not isinstance(pdata, dict):
                continue
            stats = pdata.get("learner_stats", pdata)
            if not isinstance(stats, dict):
                continue
            for k, v in stats.items():
                if not isinstance(v, numbers.Real):
                    continue
                key = str(k).lower()
                if all(n in key for n in needles):
                    vals.append(float(v))
        if not vals:
            return default
        return sum(vals) / len(vals)

    def _phase_index_for_episode(self, episode: int, phases: list[dict]) -> int:
        if not phases:
            return 0
        ep = max(0, int(episode))
        for idx, phase in enumerate(phases, start=1):
            until = int(phase.get("until_episode", 0) or 0)
            if until > 0 and ep <= until:
                return idx
        return len(phases)

    def _phase_name_for_index(self, phase_index: int, phases: list[dict]) -> str:
        if not phases or phase_index <= 0:
            return "default"
        idx = max(1, int(phase_index))
        if idx > len(phases):
            idx = len(phases)
        return str(phases[idx - 1].get("name", f"phase_{idx}"))

    def _compose_cause_histogram(
        self,
        death_histogram: Dict[str, int],
        death_by_monster_histogram: Dict[str, int],
        monster_name_map: Dict[str, str],
    ) -> Dict[str, int]:
        out: Dict[str, int] = {
            "starvation": int(death_histogram.get("starvation", 0)),
            "killed_by_monster": int(death_histogram.get("monster", 0)),
            "killed_by_agent": int(death_histogram.get("agent", 0)),
            "other": int(death_histogram.get("other", 0)),
        }
        for monster_id in sorted(death_by_monster_histogram.keys()):
            monster_name = monster_name_map.get(monster_id, monster_id)
            out[f"killed_by_monster:{monster_name}"] = int(
                death_by_monster_histogram.get(monster_id, 0)
            )
        return out

    def _load_monster_name_map(self, env_config_path: str) -> Dict[str, str]:
        try:
            raw = json.loads(Path(env_config_path).read_text(encoding="utf-8"))
            payload = raw.get("env_config", raw) if isinstance(raw, dict) else {}
            monsters_path = str(payload.get("monsters_path", "data/base/monsters.json"))
            mon_raw = json.loads(Path(monsters_path).read_text(encoding="utf-8"))
            out: Dict[str, str] = {}
            if isinstance(mon_raw, dict):
                rows = mon_raw.get("monsters", [])
                if isinstance(rows, list):
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        mid = str(row.get("monster_id", "")).strip()
                        name = str(row.get("name", "")).strip()
                        if mid:
                            out[mid] = name or mid
            return out
        except Exception:
            return {}

    def _write_metric_plots(self, out: Path, metrics_rows: list[dict]) -> Dict[str, str]:
        plots_dir = out / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "reward": [float(r.get("episode_reward_mean", 0.0) or 0.0) for r in metrics_rows],
            "rolling_reward50": [
                float(
                    sum(
                        float(metrics_rows[j].get("episode_reward_mean", 0.0) or 0.0)
                        for j in range(max(0, i - 49), i + 1)
                    )
                    / max(1, (i - max(0, i - 49) + 1))
                )
                for i in range(len(metrics_rows))
            ],
            "win_rate": [float(r.get("win_rate", 0.0) or 0.0) for r in metrics_rows],
            "survival_mean": [float(r.get("survival_mean", 0.0) or 0.0) for r in metrics_rows],
            "starvation_rate": [float(r.get("starvation_rate", 0.0) or 0.0) for r in metrics_rows],
            "loss": [float(r.get("loss", 0.0) or 0.0) for r in metrics_rows],
            "mean_teammate_distance": [
                float(r.get("mean_teammate_distance", 0.0) or 0.0) for r in metrics_rows
            ],
            "explore_coverage": [float(r.get("explore_coverage", 0.0) or 0.0) for r in metrics_rows],
            "steps_since_new_tile": [
                float(r.get("steps_since_new_tile", 0.0) or 0.0) for r in metrics_rows
            ],
            "enemy_visible_steps": [
                float(r.get("enemy_visible_steps", 0.0) or 0.0) for r in metrics_rows
            ],
            "enemy_distance": [float(r.get("enemy_distance", 0.0) or 0.0) for r in metrics_rows],
            "enemy_distance_delta_mean": [
                float(r.get("enemy_distance_delta_mean", 0.0) or 0.0) for r in metrics_rows
            ],
            "first_enemy_seen_rate": [
                float(r.get("first_enemy_seen_rate", 0.0) or 0.0) for r in metrics_rows
            ],
            "first_enemy_seen_step": [
                float(r.get("first_enemy_seen_step", 0.0) or 0.0) for r in metrics_rows
            ],
            "combat_exchanges": [float(r.get("combat_exchanges", 0.0) or 0.0) for r in metrics_rows],
            "timeout_no_contact_rate": [
                float(r.get("timeout_no_contact_rate", 0.0) or 0.0) for r in metrics_rows
            ],
            "action_wait_rate": [float(r.get("action_wait_rate", 0.0) or 0.0) for r in metrics_rows],
            "action_move_rate": [float(r.get("action_move_rate", 0.0) or 0.0) for r in metrics_rows],
            "action_interact_rate": [float(r.get("action_interact_rate", 0.0) or 0.0) for r in metrics_rows],
            "engagement_rate": [float(r.get("engagement_rate", 0.0) or 0.0) for r in metrics_rows],
            "damage_event_count": [float(r.get("damage_event_count", 0.0) or 0.0) for r in metrics_rows],
            "kill_event_count": [float(r.get("kill_event_count", 0.0) or 0.0) for r in metrics_rows],
            "reward_comp_action_total": [float(r.get("reward_comp_action_total", 0.0) or 0.0) for r in metrics_rows],
            "reward_comp_survival": [float(r.get("reward_comp_survival", 0.0) or 0.0) for r in metrics_rows],
            "reward_comp_search_explore": [float(r.get("reward_comp_search_explore", 0.0) or 0.0) for r in metrics_rows],
            "reward_comp_profile_shape": [float(r.get("reward_comp_profile_shape", 0.0) or 0.0) for r in metrics_rows],
            "reward_comp_terminal": [float(r.get("reward_comp_terminal", 0.0) or 0.0) for r in metrics_rows],
            "policy_entropy": [float(r.get("policy_entropy", 0.0) or 0.0) for r in metrics_rows],
            "policy_kl": [float(r.get("policy_kl", 0.0) or 0.0) for r in metrics_rows],
            "phase_index": [int(float(r.get("phase_index", 0) or 0)) for r in metrics_rows],
        }
        payload_path = plots_dir / "curves.json"
        payload_path.write_text(json.dumps(payload), encoding="utf-8")
        script = (
            "import json, sys\n"
            "from pathlib import Path\n"
            "import matplotlib\n"
            "matplotlib.use('Agg')\n"
            "import matplotlib.pyplot as plt\n"
            "curves = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
            "out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)\n"
            "phase_idx = curves.get('phase_index', [])\n"
            "for name, y in curves.items():\n"
            "    if name == 'phase_index':\n"
            "        continue\n"
            "    x = list(range(1, len(y)+1))\n"
            "    plt.figure(figsize=(8,3))\n"
            "    plt.plot(x, y, linewidth=1.8)\n"
            "    if phase_idx and len(phase_idx) == len(y):\n"
            "        last = phase_idx[0]\n"
            "        for i, p in enumerate(phase_idx[1:], start=2):\n"
            "            if p != last:\n"
            "                plt.axvline(i, color='#ff7f0e', linestyle='--', linewidth=1, alpha=0.8)\n"
            "                last = p\n"
            "    plt.title(name)\n"
            "    plt.xlabel('Iteration')\n"
            "    plt.grid(alpha=0.25)\n"
            "    plt.tight_layout()\n"
            "    plt.savefig(out / f'{name}.png', dpi=130)\n"
            "    plt.close()\n"
        )
        try:
            subprocess.run(
                [sys.executable, "-c", script, str(payload_path), str(plots_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return {}
        names = sorted(k for k in payload.keys() if k != "phase_index")
        return {name: f"plots/{name}.png" for name in names}

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
