"""Multi-agent trainer module for RLRLGym."""

from __future__ import annotations

import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from rlrlgym import EnvConfig, PettingZooParallelRLRLGym, TrainingLogger

from .network_config import NetworkConfig, load_network_configs
from .policies import NeuralQPolicy


@dataclass
class TrainConfig:
    episodes: int = 100
    max_steps: int = 120
    seed: int = 0
    output_dir: str = "outputs/train"
    width: int = 20
    height: int = 12
    n_agents: int = 2
    render_enabled: bool = False
    networks_path: str = "data/agent_networks.json"
    agent_profile_map: Dict[str, str] | None = None
    progress_window: int = 50
    show_progress: bool = True


class MultiAgentTrainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        profile_map = config.agent_profile_map or {"agent_0": "human", "agent_1": "orc"}
        self.env = PettingZooParallelRLRLGym(
            EnvConfig(
                width=config.width,
                height=config.height,
                max_steps=config.max_steps,
                n_agents=config.n_agents,
                render_enabled=config.render_enabled,
                agent_profile_map=profile_map,
            )
        )
        self.logger = TrainingLogger(output_dir=config.output_dir)
        self.network_cfgs: Dict[str, NetworkConfig] = load_network_configs(config.networks_path)
        self.agent_profile_map = profile_map
        self.policies: Dict[str, NeuralQPolicy] = {}
        for i, aid in enumerate(self.env.possible_agents):
            profile = self.agent_profile_map.get(aid, "human")
            if profile not in self.network_cfgs:
                raise ValueError(f"No network architecture for profile '{profile}'")
            self.policies[aid] = NeuralQPolicy(
                net_cfg=self.network_cfgs[profile],
                seed=config.seed + i,
            )

    def train(self) -> Dict[str, object]:
        window = max(1, int(self.config.progress_window))
        recent_returns: deque[float] = deque(maxlen=window)
        recent_wins: deque[float] = deque(maxlen=window)
        recent_survival: deque[float] = deque(maxlen=window)
        recent_starvation: deque[float] = deque(maxlen=window)
        recent_loss: deque[float] = deque(maxlen=window)
        recent_teammate_dist: deque[float] = deque(maxlen=window)

        for ep in range(self.config.episodes):
            observations, _ = self.env.reset(seed=self.config.seed + ep)
            self.logger.start_episode(self.env.possible_agents)
            episode_losses: list[float] = []
            episode_teammate_dist: list[float] = []

            for _ in range(self.config.max_steps):
                for aid in self.env.agents:
                    stats = observations.get(aid, {}).get("stats", {})
                    td = stats.get("teammate_distance") if isinstance(stats, dict) else None
                    if td is not None:
                        episode_teammate_dist.append(float(td))

                actions = {
                    aid: self.policies[aid].act(observations[aid], training=True)
                    for aid in self.env.agents
                }

                next_obs, rewards, terminations, truncations, info = self.env.step(actions)
                self.logger.log_step(rewards, terminations, truncations, info)

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

                observations = next_obs
                if not self.env.agents:
                    break

            alive = {aid: self.env.state.agents[aid].alive for aid in self.env.possible_agents}
            summary = self.logger.end_episode(step_count=self.env.state.step_count, alive_agents=alive)

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
                )

        if self.config.show_progress:
            sys.stdout.write("\n")
            sys.stdout.flush()

        artifact_paths = self.logger.write_outputs()
        checkpoint_path = self._write_checkpoint()
        aggregate = self.logger.aggregate_metrics()

        return {
            "aggregate": aggregate,
            "artifacts": artifact_paths,
            "checkpoint": checkpoint_path,
        }

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
        sys.stdout.write(line)
        sys.stdout.flush()

    def _write_checkpoint(self) -> str:
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        p = out / "neural_policies.json"
        payload = {aid: policy.to_dict() for aid, policy in self.policies.items()}
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(p)
