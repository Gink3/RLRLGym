"""Multi-agent trainer module for RLRLGym."""

from __future__ import annotations

import json
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
        for ep in range(self.config.episodes):
            observations, _ = self.env.reset(seed=self.config.seed + ep)
            self.logger.start_episode(self.env.possible_agents)

            for _ in range(self.config.max_steps):
                actions = {
                    aid: self.policies[aid].act(observations[aid], training=True)
                    for aid in self.env.agents
                }

                next_obs, rewards, terminations, truncations, info = self.env.step(actions)
                self.logger.log_step(rewards, terminations, truncations, info)

                for aid, action in actions.items():
                    done = bool(terminations.get(aid, False) or truncations.get(aid, False))
                    self.policies[aid].update(
                        observation=observations[aid],
                        action=action,
                        reward=float(rewards.get(aid, 0.0)),
                        next_observation=next_obs.get(aid),
                        done=done,
                    )

                observations = next_obs
                if not self.env.agents:
                    break

            alive = {aid: self.env.state.agents[aid].alive for aid in self.env.possible_agents}
            self.logger.end_episode(step_count=self.env.state.step_count, alive_agents=alive)

            for policy in self.policies.values():
                policy.decay_epsilon()

        artifact_paths = self.logger.write_outputs()
        checkpoint_path = self._write_checkpoint()
        aggregate = self.logger.aggregate_metrics()

        return {
            "aggregate": aggregate,
            "artifacts": artifact_paths,
            "checkpoint": checkpoint_path,
        }

    def _write_checkpoint(self) -> str:
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        p = out / "neural_policies.json"
        payload = {aid: policy.to_dict() for aid, policy in self.policies.items()}
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(p)
