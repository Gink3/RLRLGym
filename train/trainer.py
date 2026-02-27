"""Multi-agent trainer module for RLRLGym."""

from __future__ import annotations

import json
import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from rlrlgym import EnvConfig, PettingZooParallelRLRLGym
from rlrlgym.constants import ACTION_NAMES
from rlrlgym.scenario import estimate_max_networks

from .aim_logger import AimLogger
from .network_config import NetworkConfig, load_network_configs
from .policies import NeuralQPolicy

EXPLORER_PROFILES = {"reward_explorer_policy_v1", "human"}
BRAWLER_PROFILES = {"reward_brawler_policy_v1", "orc"}


@dataclass
class TrainConfig:
    episodes: int = 100
    max_steps: Optional[int] = None
    seed: int = 0
    output_dir: str = "outputs/train"
    width: Optional[int] = None
    height: Optional[int] = None
    n_agents: Optional[int] = None
    render_enabled: bool = False
    networks_path: str = "data/base/agent_networks.json"
    agent_profile_map: Dict[str, str] | None = None
    progress_window: int = 50
    show_progress: bool = True
    replay_save_every: int = 5000
    env_config_path: str = "data/env_config.json"
    scenario_path: str = ""
    resource_guard_enabled: bool = True
    resource_guard_ram_fraction: float = 0.45
    resource_guard_bytes_per_param: int = 32
    max_nn_policies: int = 0
    aim_enabled: bool = True
    aim_experiment: str = "rlrlgym_custom"
    aim_repo_path: str = "/proj/aimml"


@dataclass
class EpisodeSummary:
    episode: int
    steps: int
    team_return: float
    per_agent_return: Dict[str, float]
    win: bool
    outcome: str
    mean_survival_time: float
    cause_of_death: Dict[str, str]
    action_counts: Dict[str, int]


class MultiAgentTrainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        profile_map = config.agent_profile_map or {}
        env_cfg = EnvConfig.from_json(config.env_config_path)
        if config.width is not None:
            env_cfg.width = int(config.width)
        if config.height is not None:
            env_cfg.height = int(config.height)
        if config.max_steps is not None:
            env_cfg.max_steps = int(config.max_steps)
        if config.n_agents is not None:
            env_cfg.n_agents = int(config.n_agents)
        env_cfg.render_enabled = config.render_enabled
        if config.scenario_path:
            env_cfg.scenario_path = str(config.scenario_path)
        elif profile_map:
            env_cfg.agent_profile_map = dict(profile_map)
        self.env = PettingZooParallelRLRLGym(
            env_cfg
        )
        self.network_cfgs: Dict[str, NetworkConfig] = load_network_configs(config.networks_path)
        self.agent_profile_map = dict(self.env.config.agent_profile_map)
        self.agent_network_map: Dict[str, str] = {}
        self.policies: Dict[str, NeuralQPolicy] = {}
        for i, aid in enumerate(self.env.possible_agents):
            network_name = self._resolve_network_name(aid)
            if network_name not in self.network_cfgs:
                raise ValueError(f"No network architecture for '{network_name}'")
            self.agent_network_map[aid] = network_name
            self.policies[aid] = NeuralQPolicy(
                net_cfg=self.network_cfgs[network_name],
                seed=config.seed + i,
            )
        self._run_metrics_initialized = False
        self.aim = AimLogger(
            enabled=config.aim_enabled,
            experiment=config.aim_experiment,
            repo_path=config.aim_repo_path,
            run_name="custom_trainer",
        )
        self.aim.set_params(
            {
                "backend": "custom",
                "episodes": int(config.episodes),
                "seed": int(config.seed),
                "output_dir": str(config.output_dir),
                "width": None if config.width is None else int(config.width),
                "height": None if config.height is None else int(config.height),
                "n_agents": None if config.n_agents is None else int(config.n_agents),
                "max_steps": None if config.max_steps is None else int(config.max_steps),
                "networks_path": str(config.networks_path),
                "env_config_path": str(config.env_config_path),
                "scenario_path": str(config.scenario_path or ""),
                "aim_repo_path": str(config.aim_repo_path),
            }
        )

    def _resolve_network_name(self, aid: str) -> str:
        idx = self.env.possible_agents.index(aid)
        scenario_row: Dict[str, object] = {}
        scenario = getattr(self.env.config, "agent_scenario", [])
        if isinstance(scenario, list) and 0 <= idx < len(scenario):
            row = scenario[idx]
            if isinstance(row, dict):
                scenario_row = row
        explicit = str(scenario_row.get("network", "")).strip()
        if explicit and explicit in self.network_cfgs:
            return explicit
        race = str(self.env.config.agent_race_map.get(aid, "")).strip()
        cls = str(self.env.config.agent_class_map.get(aid, "")).strip()
        profile = str(self.env.config.agent_profile_map.get(aid, "")).strip()
        candidates = [
            f"{race}_{cls}" if race and cls else "",
            profile,
            race,
            "default",
        ]
        for name in candidates:
            if name and name in self.network_cfgs:
                return name
        return sorted(self.network_cfgs.keys())[0]

    def _available_ram_bytes(self) -> int:
        try:
            page = int(os.sysconf("SC_PAGE_SIZE"))
            avail = int(os.sysconf("SC_AVPHYS_PAGES"))
            return max(0, page * avail)
        except Exception:
            return 0

    def train(self) -> Dict[str, object]:
        window = max(1, int(self.config.progress_window))
        recent_returns: deque[float] = deque(maxlen=window)
        recent_wins: deque[float] = deque(maxlen=window)
        recent_survival: deque[float] = deque(maxlen=window)
        recent_starvation: deque[float] = deque(maxlen=window)
        recent_loss: deque[float] = deque(maxlen=window)
        recent_teammate_dist: deque[float] = deque(maxlen=window)
        recent_profile_metrics: Dict[str, Dict[str, deque[float]]] = {}
        replay_paths: list[str] = []
        episode_summaries: list[EpisodeSummary] = []

        for ep in range(self.config.episodes):
            observations, _ = self.env.reset(seed=self.config.seed + ep)
            capture_replay = (
                int(self.config.replay_save_every) > 0
                and (ep + 1) % int(self.config.replay_save_every) == 0
            )
            replay_states = (
                [self.env.capture_playback_state()] if capture_replay else []
            )
            replay_actions = [] if capture_replay else None
            replay_step_logs = [] if capture_replay else None
            if not self._run_metrics_initialized:
                self._initialize_run_metrics(observations)
            episode_losses: list[float] = []
            episode_teammate_dist: list[float] = []
            episode_agent_returns = {aid: 0.0 for aid in self.env.possible_agents}
            episode_agent_losses = {aid: [] for aid in self.env.possible_agents}
            episode_agent_dist = {aid: [] for aid in self.env.possible_agents}
            episode_agent_survival = {aid: 0 for aid in self.env.possible_agents}
            episode_agent_done = {aid: False for aid in self.env.possible_agents}
            episode_death_cause = {aid: "alive" for aid in self.env.possible_agents}
            episode_action_counts = {
                ACTION_NAMES[k]: 0 for k in sorted(ACTION_NAMES.keys())
            }

            for _ in range(self.env.config.max_steps):
                for aid in self.env.possible_agents:
                    if not episode_agent_done[aid]:
                        episode_agent_survival[aid] += 1

                for aid in self.env.agents:
                    stats = observations.get(aid, {}).get("stats", {})
                    td = stats.get("teammate_distance") if isinstance(stats, dict) else None
                    if td is not None:
                        episode_teammate_dist.append(float(td))
                        episode_agent_dist[aid].append(float(td))

                actions = {
                    aid: self.policies[aid].act(observations[aid], training=True)
                    for aid in self.env.agents
                }

                next_obs, rewards, terminations, truncations, info = self.env.step(actions)
                if capture_replay:
                    replay_states.append(self.env.capture_playback_state())
                    assert replay_actions is not None
                    replay_actions.append({aid: int(a) for aid, a in actions.items()})
                    assert replay_step_logs is not None
                    prev_state = replay_states[-2] if len(replay_states) >= 2 else None
                    curr_state = replay_states[-1]
                    replay_step_logs.append(
                        self._build_replay_step_log(
                            actions=actions,
                            rewards=rewards,
                            terminations=terminations,
                            truncations=truncations,
                            info=info,
                            prev_state=prev_state,
                            curr_state=curr_state,
                        )
                    )
                for aid, reward in rewards.items():
                    episode_agent_returns[aid] += float(reward)
                for aid in self.env.possible_agents:
                    if terminations.get(aid, False) or truncations.get(aid, False):
                        episode_agent_done[aid] = True
                        if episode_death_cause[aid] == "alive":
                            events = list(info.get(aid, {}).get("events", []))
                            if "death" in events and "starve_tick" in events:
                                episode_death_cause[aid] = "starvation"
                            elif "death" in events:
                                episode_death_cause[aid] = "damage_or_other"
                            elif truncations.get(aid, False):
                                episode_death_cause[aid] = "timeout_alive"
                            else:
                                episode_death_cause[aid] = "unknown"
                for action in actions.values():
                    name = ACTION_NAMES.get(int(action), f"unknown_{int(action)}")
                    episode_action_counts[name] = episode_action_counts.get(name, 0) + 1

                for aid, action in actions.items():
                    done = bool(terminations.get(aid, False) or truncations.get(aid, False))
                    loss = self.policies[aid].update(
                        observation=observations[aid],
                        action=action,
                        reward=float(rewards.get(aid, 0.0)),
                        next_observation=next_obs.get(aid),
                        done=done,
                    )
                    episode_losses.append(float(loss))
                    episode_agent_losses[aid].append(float(loss))

                observations = next_obs
                if not self.env.agents:
                    break

            alive = {aid: self.env.state.agents[aid].alive for aid in self.env.possible_agents}
            explorer_alive = any(
                bool(alive.get(aid, False))
                for aid in self.env.possible_agents
                if self.env.state.agents[aid].profile_name in EXPLORER_PROFILES
            )
            brawler_alive = any(
                bool(alive.get(aid, False))
                for aid in self.env.possible_agents
                if self.env.state.agents[aid].profile_name in BRAWLER_PROFILES
            )
            if explorer_alive and not brawler_alive:
                outcome = "explorer_win"
            elif brawler_alive and not explorer_alive:
                outcome = "brawler_win"
            else:
                outcome = "tie"
            summary = EpisodeSummary(
                episode=ep + 1,
                steps=int(self.env.state.step_count),
                team_return=float(sum(episode_agent_returns.values())),
                per_agent_return=dict(episode_agent_returns),
                win=any(alive.values()),
                outcome=outcome,
                mean_survival_time=(
                    sum(float(v) for v in episode_agent_survival.values())
                    / max(1, len(episode_agent_survival))
                ),
                cause_of_death=dict(episode_death_cause),
                action_counts=dict(episode_action_counts),
            )
            episode_summaries.append(summary)
            if capture_replay:
                replay_paths.append(
                    self._write_replay(
                        episode=ep + 1,
                        seed=self.config.seed + ep,
                        frames=replay_states,
                        action_history=replay_actions or [],
                        step_logs=replay_step_logs or [],
                    )
                )

            for policy in self.policies.values():
                policy.decay_epsilon()

            episode_mean_loss = sum(episode_losses) / max(1, len(episode_losses))
            episode_mean_teammate = (
                sum(episode_teammate_dist) / len(episode_teammate_dist)
                if episode_teammate_dist
                else 0.0
            )
            episode_starved = (
                1.0
                if any(cause == "starvation" for cause in summary.cause_of_death.values())
                else 0.0
            )

            recent_returns.append(float(summary.team_return))
            recent_wins.append(1.0 if summary.win else 0.0)
            recent_survival.append(float(summary.mean_survival_time))
            recent_starvation.append(episode_starved)
            recent_loss.append(float(episode_mean_loss))
            recent_teammate_dist.append(float(episode_mean_teammate))
            self._update_profile_metric_windows(
                recent_profile_metrics=recent_profile_metrics,
                episode_returns=episode_agent_returns,
                episode_losses=episode_agent_losses,
                episode_survival=episode_agent_survival,
                episode_teammate_dist=episode_agent_dist,
                cause_of_death=summary.cause_of_death,
                alive_agents=alive,
                window=window,
            )

            if self.config.show_progress:
                self._print_live_progress(
                    episode=ep + 1,
                    total=self.config.episodes,
                    window=window,
                    moving_avg_team_return=sum(recent_returns) / len(recent_returns),
                    moving_win_rate=sum(recent_wins) / len(recent_wins),
                    moving_mean_survival_steps=sum(recent_survival) / len(recent_survival),
                    starvation_rate=sum(recent_starvation) / len(recent_starvation),
                    mean_loss=sum(recent_loss) / len(recent_loss),
                    epsilon=sum(p.epsilon for p in self.policies.values()) / len(self.policies),
                    mean_teammate_distance=sum(recent_teammate_dist) / len(recent_teammate_dist),
                    profile_metrics=recent_profile_metrics,
                )
            self._log_aim_episode_metrics(
                episode=ep + 1,
                summary=summary,
                episode_mean_loss=episode_mean_loss,
                episode_mean_teammate=episode_mean_teammate,
                starvation_rate=episode_starved,
                recent_returns=recent_returns,
                recent_wins=recent_wins,
                recent_survival=recent_survival,
                recent_starvation=recent_starvation,
                recent_loss=recent_loss,
                recent_teammate_dist=recent_teammate_dist,
                recent_profile_metrics=recent_profile_metrics,
                episode_summaries=episode_summaries,
                window=window,
            )
        if self.config.show_progress:
            sys.stdout.write("\n")
            sys.stdout.flush()

        checkpoint_path = self._write_checkpoint()
        aggregate = self._aggregate_episode_summaries(episode_summaries)
        self.aim.set_payload("aggregate/final", aggregate)
        self.aim.close()

        return {
            "aggregate": aggregate,
            "artifacts": {},
            "checkpoint": checkpoint_path,
            "replays": replay_paths,
        }

    def _log_aim_episode_metrics(
        self,
        episode: int,
        summary,
        episode_mean_loss: float,
        episode_mean_teammate: float,
        starvation_rate: float,
        recent_returns,
        recent_wins,
        recent_survival,
        recent_starvation,
        recent_loss,
        recent_teammate_dist,
        recent_profile_metrics,
        episode_summaries,
        window: int,
    ) -> None:
        self.aim.track_pairs(
            [
                ("steps", summary.steps),
                ("team_return", summary.team_return),
                ("win", summary.win),
                ("outcome_explorer_win", summary.outcome == "explorer_win"),
                ("outcome_brawler_win", summary.outcome == "brawler_win"),
                ("outcome_tie", summary.outcome == "tie"),
                ("mean_survival_time", summary.mean_survival_time),
                ("episode_mean_loss", episode_mean_loss),
                ("episode_mean_teammate_distance", episode_mean_teammate),
                ("episode_starvation", starvation_rate),
                ("rolling_return_window", sum(recent_returns) / max(1, len(recent_returns))),
                ("rolling_win_window", sum(recent_wins) / max(1, len(recent_wins))),
                ("rolling_survival_window", sum(recent_survival) / max(1, len(recent_survival))),
                ("rolling_starvation_window", sum(recent_starvation) / max(1, len(recent_starvation))),
                ("rolling_loss_window", sum(recent_loss) / max(1, len(recent_loss))),
                (
                    "rolling_teammate_distance_window",
                    sum(recent_teammate_dist) / max(1, len(recent_teammate_dist)),
                ),
                ("rolling_window_size", window),
            ],
            step=episode,
            prefix="custom/episode",
        )
        self.aim.track_many(
            summary.per_agent_return,
            step=episode,
            prefix="custom/per_agent_return",
        )
        self.aim.track_many(
            summary.action_counts,
            step=episode,
            prefix="custom/action_counts",
        )
        cod_counts: Dict[str, int] = {}
        for cause in summary.cause_of_death.values():
            cod_counts[cause] = cod_counts.get(cause, 0) + 1
        self.aim.track_many(
            cod_counts,
            step=episode,
            prefix="custom/cause_of_death",
        )
        for profile, metrics in recent_profile_metrics.items():
            self.aim.track_pairs(
                [
                    ("return", self._mean(metrics["return"])),
                    ("win", self._mean(metrics["win"])),
                    ("survival", self._mean(metrics["survival"])),
                    ("starvation", self._mean(metrics["starvation"])),
                    ("loss", self._mean(metrics["loss"])),
                    ("distance", self._mean(metrics["distance"])),
                ],
                step=episode,
                prefix=f"custom/profile/{profile}",
            )
        aggregate = self._aggregate_episode_summaries(episode_summaries)
        self.aim.track_pairs(
            [
                ("episodes", aggregate["episodes"]),
                ("win_rate", aggregate["win_rate"]),
                ("human_win_rate", aggregate["human_win_rate"]),
                ("orc_win_rate", aggregate["orc_win_rate"]),
                ("tie_rate", aggregate["tie_rate"]),
                ("mean_team_return", aggregate["mean_team_return"]),
                ("mean_survival_time", aggregate["mean_survival_time"]),
            ],
            step=episode,
            prefix="custom/aggregate",
        )
        self.aim.track_many(
            aggregate.get("cause_of_death_histogram", {}),
            step=episode,
            prefix="custom/aggregate/cause_of_death",
        )
        self.aim.track_many(
            aggregate.get("action_histogram", {}),
            step=episode,
            prefix="custom/aggregate/action_histogram",
        )

    def _aggregate_episode_summaries(
        self, episode_summaries: list[EpisodeSummary]
    ) -> Dict[str, object]:
        if not episode_summaries:
            return {
                "episodes": 0,
                "win_rate": 0.0,
                "explorer_win_rate": 0.0,
                "brawler_win_rate": 0.0,
                "human_win_rate": 0.0,
                "orc_win_rate": 0.0,
                "tie_rate": 0.0,
                "mean_team_return": 0.0,
                "mean_survival_time": 0.0,
                "cause_of_death_histogram": {},
                "action_histogram": {},
                "run_metrics": {
                    "network_parameter_counts": self._network_parameter_counts(),
                },
            }
        wins = sum(1 for e in episode_summaries if e.win)
        explorer_wins = sum(1 for e in episode_summaries if e.outcome == "explorer_win")
        brawler_wins = sum(1 for e in episode_summaries if e.outcome == "brawler_win")
        ties = sum(1 for e in episode_summaries if e.outcome == "tie")
        cod: Dict[str, int] = {}
        actions: Dict[str, int] = {}
        for e in episode_summaries:
            for cause in e.cause_of_death.values():
                cod[cause] = cod.get(cause, 0) + 1
            for action_name, count in e.action_counts.items():
                actions[action_name] = actions.get(action_name, 0) + int(count)
        n = len(episode_summaries)
        return {
            "episodes": n,
            "win_rate": wins / n,
            "explorer_win_rate": explorer_wins / n,
            "brawler_win_rate": brawler_wins / n,
            # Legacy keys retained for backward-compatible dashboards.
            "human_win_rate": explorer_wins / n,
            "orc_win_rate": brawler_wins / n,
            "tie_rate": ties / n,
            "mean_team_return": sum(e.team_return for e in episode_summaries) / n,
            "mean_survival_time": sum(e.mean_survival_time for e in episode_summaries) / n,
            "cause_of_death_histogram": cod,
            "action_histogram": actions,
            "run_metrics": {
                "network_parameter_counts": self._network_parameter_counts(),
            },
        }

    def _network_parameter_counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for aid in self.env.possible_agents:
            out[aid] = self.policies[aid].parameter_count()
        return out

    def _initialize_run_metrics(self, observations: Dict[str, Dict[str, object]]) -> None:
        # Build policy networks once from first observations, then capture static size metadata.
        for aid in self.env.possible_agents:
            self.policies[aid].ensure_initialized(observations[aid])

        per_agent_param_counts: Dict[str, int] = {}
        for aid in self.env.possible_agents:
            per_agent_param_counts[aid] = self.policies[aid].parameter_count()

        ram_avail = self._available_ram_bytes()
        peak_params = max(1, max(per_agent_param_counts.values(), default=1))
        if ram_avail > 0:
            est_max_networks, est_bytes_per_network = estimate_max_networks(
                per_network_params=peak_params,
                available_ram_bytes=ram_avail,
                usable_fraction=float(self.config.resource_guard_ram_fraction),
                bytes_per_param=int(self.config.resource_guard_bytes_per_param),
            )
        else:
            est_max_networks = len(self.policies)
            est_bytes_per_network = peak_params * int(
                max(8, int(self.config.resource_guard_bytes_per_param))
            )
        configured_cap = int(self.config.max_nn_policies)
        hard_cap = configured_cap if configured_cap > 0 else est_max_networks
        if bool(self.config.resource_guard_enabled) and len(self.policies) > hard_cap:
            raise RuntimeError(
                "Scenario exceeds estimated NN capacity: "
                f"{len(self.policies)} policies requested, cap={hard_cap}. "
                "Reduce agents, shrink network hidden layers, or increase system memory."
            )

        self.aim.set_payload(
            "custom/run_metrics",
            {
                "network_parameter_counts": per_agent_param_counts,
                "network_arch_by_agent": dict(self.agent_network_map),
                "estimated_bytes_per_network": int(est_bytes_per_network),
                "estimated_max_networks_from_ram": int(est_max_networks),
                "available_ram_bytes": int(ram_avail),
                "active_network_count": int(len(self.policies)),
                "resource_guard_hard_cap": int(hard_cap),
            },
        )
        self._run_metrics_initialized = True

    def _print_live_progress(
        self,
        episode: int,
        total: int,
        window: int,
        moving_avg_team_return: float,
        moving_win_rate: float,
        moving_mean_survival_steps: float,
        starvation_rate: float,
        mean_loss: float,
        epsilon: float,
        mean_teammate_distance: float,
        profile_metrics: Dict[str, Dict[str, deque[float]]],
    ) -> None:
        bar_width = 26
        frac = episode / max(1, total)
        filled = int(bar_width * frac)
        bar = "#" * filled + "-" * (bar_width - filled)
        line = (
            f"\r[{bar}] {episode}/{total} "
            f"ret{window}={moving_avg_team_return:.3f} "
            f"win{window}={moving_win_rate:.3f} "
            f"surv{window}={moving_mean_survival_steps:.1f} "
            f"starve{window}={starvation_rate:.3f} "
            f"loss{window}={mean_loss:.4f} "
            f"eps={epsilon:.3f} "
            f"team_dist{window}={mean_teammate_distance:.2f}"
        )
        if profile_metrics:
            line += " |"
            for profile in sorted(profile_metrics.keys()):
                pm = profile_metrics[profile]
                pret = self._mean(pm["return"])
                pwin = self._mean(pm["win"])
                psurv = self._mean(pm["survival"])
                pstarve = self._mean(pm["starvation"])
                ploss = self._mean(pm["loss"])
                pdist = self._mean(pm["distance"])
                line += (
                    f" {profile}:"
                    f"ret={pret:.3f},"
                    f"win={pwin:.3f},"
                    f"surv={psurv:.1f},"
                    f"starve={pstarve:.3f},"
                    f"loss={ploss:.4f},"
                    f"dist={pdist:.2f}"
                )
        sys.stdout.write(line)
        sys.stdout.flush()

    def _update_profile_metric_windows(
        self,
        recent_profile_metrics: Dict[str, Dict[str, deque[float]]],
        episode_returns: Dict[str, float],
        episode_losses: Dict[str, list[float]],
        episode_survival: Dict[str, int],
        episode_teammate_dist: Dict[str, list[float]],
        cause_of_death: Dict[str, str],
        alive_agents: Dict[str, bool],
        window: int,
    ) -> None:
        profile_to_agent_ids: Dict[str, list[str]] = {}
        for aid in self.env.possible_agents:
            profile = self.env.state.agents[aid].profile_name
            profile_to_agent_ids.setdefault(profile, []).append(aid)

        for profile, aids in profile_to_agent_ids.items():
            if profile not in recent_profile_metrics:
                recent_profile_metrics[profile] = {
                    "return": deque(maxlen=window),
                    "win": deque(maxlen=window),
                    "survival": deque(maxlen=window),
                    "starvation": deque(maxlen=window),
                    "loss": deque(maxlen=window),
                    "distance": deque(maxlen=window),
                }
            pm = recent_profile_metrics[profile]

            returns = [episode_returns[aid] for aid in aids]
            wins = [1.0 if alive_agents.get(aid, False) else 0.0 for aid in aids]
            survival = [float(episode_survival[aid]) for aid in aids]
            starvation = [
                1.0 if cause_of_death.get(aid) == "starvation" else 0.0 for aid in aids
            ]
            losses = [
                (sum(episode_losses[aid]) / len(episode_losses[aid]))
                if episode_losses[aid]
                else 0.0
                for aid in aids
            ]
            distances = [
                (sum(episode_teammate_dist[aid]) / len(episode_teammate_dist[aid]))
                if episode_teammate_dist[aid]
                else 0.0
                for aid in aids
            ]

            pm["return"].append(sum(returns) / max(1, len(returns)))
            pm["win"].append(sum(wins) / max(1, len(wins)))
            pm["survival"].append(sum(survival) / max(1, len(survival)))
            pm["starvation"].append(sum(starvation) / max(1, len(starvation)))
            pm["loss"].append(sum(losses) / max(1, len(losses)))
            pm["distance"].append(sum(distances) / max(1, len(distances)))

    def _mean(self, values: deque[float]) -> float:
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _write_checkpoint(self) -> str:
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        p = out / "neural_policies.json"
        payload = {aid: policy.to_dict() for aid, policy in self.policies.items()}
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(p)

    def _write_replay(
        self,
        episode: int,
        seed: int,
        frames: list,
        action_history: list[Dict[str, int]],
        step_logs: list[Dict[str, object]] | None = None,
    ) -> str:
        out = Path(self.config.output_dir) / "replays"
        out.mkdir(parents=True, exist_ok=True)
        p = out / f"episode_{episode:06d}.replay.json"
        payload = {
            "schema_version": 1,
            "episode": int(episode),
            "seed": int(seed),
            "frame_count": len(frames),
            "frames": [self._serialize_state(s) for s in frames],
            "actions": [{aid: int(a) for aid, a in x.items()} for x in action_history],
            "step_logs": list(step_logs or []),
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(p)

    def _build_replay_step_log(
        self,
        actions: Dict[str, int],
        rewards: Dict[str, float],
        terminations: Dict[str, bool],
        truncations: Dict[str, bool],
        info: Dict[str, Dict[str, object]],
        prev_state=None,
        curr_state=None,
    ) -> Dict[str, object]:
        derived_death_reasons: Dict[str, str] = {}
        for source_aid, agent_info in info.items():
            if not isinstance(agent_info, dict):
                continue
            events = list(agent_info.get("events", []))
            for evt in events:
                if isinstance(evt, str) and evt.startswith("agent_interact:kill:"):
                    victim = evt.split(":", 2)[2]
                    derived_death_reasons[victim] = f"killed by agent ({source_aid})"
        logs: Dict[str, Dict[str, object]] = {}
        for aid in self.env.possible_agents:
            agent_info = info.get(aid, {})
            events = list(agent_info.get("events", [])) if isinstance(agent_info, dict) else []
            reason = self._death_reason_from_events(events)
            if reason is None:
                reason = derived_death_reasons.get(aid)
            if reason is None and bool(terminations.get(aid, False)):
                reason = "terminated/unknown"
            logs[aid] = {
                "action": int(actions.get(aid, -1)),
                "reward": float(rewards.get(aid, 0.0)),
                "events": events,
                "terminated": bool(terminations.get(aid, False)),
                "truncated": bool(truncations.get(aid, False)),
                "death_reason": reason,
                "winner": any(str(evt) == f"winner:{aid}" for evt in events),
            }
        payload: Dict[str, object] = {"agents": logs}
        if prev_state is not None and curr_state is not None:
            payload["agent_damage"] = self._agent_damage_events(
                prev_state=prev_state, curr_state=curr_state, info=info
            )
            payload["monster_damage"] = self._monster_damage_events(
                prev_state=prev_state, curr_state=curr_state, info=info
            )
            payload["monster_deaths"] = self._monster_death_events(
                prev_state=prev_state, curr_state=curr_state, info=info
            )
        return payload

    def _death_reason_from_events(self, events: list[str]) -> str | None:
        if not events:
            return None
        for evt in events:
            if evt.startswith("death_by_monster:"):
                monster = evt.split(":", 1)[1]
                return f"killed by monster ({monster})"
        if "death" in events:
            if "starve_tick" in events:
                return "starvation"
            return "combat/unknown"
        return None

    def _agent_damage_events(self, prev_state, curr_state, info) -> list[dict]:
        out = []
        for aid, curr_agent in curr_state.agents.items():
            prev_agent = prev_state.agents.get(aid)
            if prev_agent is None:
                continue
            dmg = int(prev_agent.hp) - int(curr_agent.hp)
            if dmg <= 0:
                continue
            source = "unknown"
            events = list(info.get(aid, {}).get("events", []))
            if "starve_tick" in events:
                source = "starvation"
            else:
                for evt in events:
                    if isinstance(evt, str) and evt.startswith("death_by_monster:"):
                        source = evt.split(":", 1)[1]
                        break
                    if isinstance(evt, str) and evt.startswith("monster_hit:"):
                        parts = evt.split(":")
                        if len(parts) >= 2:
                            source = f"monster:{parts[1]}"
                            break
                if source == "unknown":
                    for src_aid, src_info in info.items():
                        for evt in list(src_info.get("events", [])):
                            if isinstance(evt, str) and evt == f"agent_interact:hit:{aid}":
                                source = f"agent:{src_aid}"
                                break
                        if source != "unknown":
                            break
            out.append({"agent_id": aid, "amount": dmg, "source": source})
        return out

    def _monster_death_events(self, prev_state, curr_state, info) -> list[dict]:
        out = []
        killer_by_monster_id: Dict[str, str] = {}
        for src_aid, src_info in info.items():
            for evt in list(src_info.get("events", [])):
                if isinstance(evt, str) and evt.startswith("agent_interact:kill_monster:"):
                    monster_id = evt.split(":", 2)[2]
                    killer_by_monster_id[monster_id] = src_aid
        for entity_id, curr_mon in curr_state.monsters.items():
            prev_mon = prev_state.monsters.get(entity_id)
            if prev_mon is None:
                continue
            if bool(prev_mon.alive) and not bool(curr_mon.alive):
                killer = killer_by_monster_id.get(curr_mon.monster_id)
                reason = f"killed by agent ({killer})" if killer else "died/unknown"
                out.append(
                    {
                        "entity_id": entity_id,
                        "monster_id": curr_mon.monster_id,
                        "reason": reason,
                    }
                )
        return out

    def _monster_damage_events(self, prev_state, curr_state, info) -> list[dict]:
        out = []
        for entity_id, curr_mon in curr_state.monsters.items():
            prev_mon = prev_state.monsters.get(entity_id)
            if prev_mon is None:
                continue
            dmg = int(prev_mon.hp) - int(curr_mon.hp)
            if dmg <= 0:
                continue
            source = "unknown"
            for src_aid, src_info in info.items():
                for evt in list(src_info.get("events", [])):
                    if (
                        isinstance(evt, str)
                        and evt == f"agent_interact:hit_monster:{entity_id}"
                    ):
                        source = f"agent:{src_aid}"
                        break
                if source != "unknown":
                    break
            out.append(
                {
                    "entity_id": entity_id,
                    "monster_id": curr_mon.monster_id,
                    "amount": dmg,
                    "hp_before": int(prev_mon.hp),
                    "hp_after": int(curr_mon.hp),
                    "hp_max": int(curr_mon.max_hp),
                    "source": source,
                }
            )
        return out

    def _serialize_state(self, state) -> Dict[str, object]:
        return {
            "grid": state.grid,
            "tile_interactions": [
                {"position": [r, c], "count": count}
                for (r, c), count in sorted(state.tile_interactions.items())
            ],
            "ground_items": [
                {"position": [r, c], "items": list(items)}
                for (r, c), items in sorted(state.ground_items.items())
            ],
            "chests": [
                {
                    "position": [r, c],
                    "opened": bool(chest.opened),
                    "locked": bool(chest.locked),
                    "loot": list(chest.loot),
                }
                for (r, c), chest in sorted(state.chests.items())
            ],
            "factions": {
                "leaders": {
                    str(fid): str(leader)
                    for fid, leader in sorted(state.faction_leaders.items())
                },
                "pending_invites": {
                    str(aid): {
                        "faction_id": int(invite.get("faction_id", -1)),
                        "inviter_id": str(invite.get("inviter_id", "")),
                        "created_step": int(invite.get("created_step", -1)),
                    }
                    for aid, invite in sorted(state.pending_faction_invites.items())
                },
            },
            "monsters": [
                {
                    "entity_id": monster.entity_id,
                    "monster_id": monster.monster_id,
                    "name": monster.name,
                    "symbol": monster.symbol,
                    "color": monster.color,
                    "position": [monster.position[0], monster.position[1]],
                    "hp": monster.hp,
                    "max_hp": monster.max_hp,
                    "acc": monster.acc,
                    "eva": monster.eva,
                    "dmg_min": monster.dmg_min,
                    "dmg_max": monster.dmg_max,
                    "dr_min": monster.dr_min,
                    "dr_max": monster.dr_max,
                    "alive": bool(monster.alive),
                }
                for _, monster in sorted(state.monsters.items())
            ],
            "agents": {
                aid: {
                    "agent_id": agent.agent_id,
                    "position": [agent.position[0], agent.position[1]],
                    "profile_name": agent.profile_name,
                    "race_name": agent.race_name,
                    "class_name": agent.class_name,
                    "hp": agent.hp,
                    "max_hp": agent.max_hp,
                    "hunger": agent.hunger,
                    "max_hunger": agent.max_hunger,
                    "inventory": list(agent.inventory),
                    "equipped": list(agent.equipped),
                    "armor_slots": dict(agent.armor_slots),
                    "faction_id": int(agent.faction_id),
                    "alive": agent.alive,
                    "visited": [
                        [r, c] for (r, c) in sorted(agent.visited)
                    ],
                    "wait_streak": agent.wait_streak,
                    "recent_positions": [
                        [r, c] for (r, c) in agent.recent_positions
                    ],
                    "strength": agent.strength,
                    "dexterity": agent.dexterity,
                    "intellect": agent.intellect,
                    "skills": dict(agent.skills),
                    "skill_xp": dict(agent.skill_xp),
                }
                for aid, agent in sorted(state.agents.items())
            },
            "step_count": state.step_count,
        }
