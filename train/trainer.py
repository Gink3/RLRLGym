"""Multi-agent trainer module for RLRLGym."""

from __future__ import annotations

import json
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from rlrlgym import EnvConfig, PettingZooParallelRLRLGym, TrainingLogger

from .network_config import NetworkConfig, load_network_configs
from .policies import NeuralQPolicy


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
    networks_path: str = "data/agent_networks.json"
    agent_profile_map: Dict[str, str] | None = None
    progress_window: int = 50
    show_progress: bool = True
    replay_save_every: int = 1000
    env_config_path: str = "data/env_config.json"


class MultiAgentTrainer:
    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        profile_map = config.agent_profile_map or {"agent_0": "human", "agent_1": "orc"}
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
        env_cfg.agent_profile_map = dict(profile_map)
        self.env = PettingZooParallelRLRLGym(
            env_cfg
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
        self._run_metrics_initialized = False

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
            self.logger.start_episode(
                self.env.possible_agents,
                agent_profiles={aid: self.env.state.agents[aid].profile_name for aid in self.env.possible_agents},
            )
            episode_losses: list[float] = []
            episode_teammate_dist: list[float] = []
            episode_agent_returns = {aid: 0.0 for aid in self.env.possible_agents}
            episode_agent_losses = {aid: [] for aid in self.env.possible_agents}
            episode_agent_dist = {aid: [] for aid in self.env.possible_agents}
            episode_agent_survival = {aid: 0 for aid in self.env.possible_agents}
            episode_agent_done = {aid: False for aid in self.env.possible_agents}

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
                self.logger.log_step(
                    rewards,
                    terminations,
                    truncations,
                    info,
                    actions=actions,
                )
                for aid, reward in rewards.items():
                    episode_agent_returns[aid] += float(reward)
                for aid in self.env.possible_agents:
                    if terminations.get(aid, False) or truncations.get(aid, False):
                        episode_agent_done[aid] = True

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
            summary = self.logger.end_episode(step_count=self.env.state.step_count, alive_agents=alive)
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
            "replays": replay_paths,
        }

    def _initialize_run_metrics(self, observations: Dict[str, Dict[str, object]]) -> None:
        # Build policy networks once from first observations, then capture static size metadata.
        for aid in self.env.possible_agents:
            self.policies[aid].ensure_initialized(observations[aid])

        profile_param_counts: Dict[str, int] = {}
        for aid in self.env.possible_agents:
            profile = self.env.state.agents[aid].profile_name
            if profile in profile_param_counts:
                continue
            profile_param_counts[profile] = self.policies[aid].parameter_count()

        self.logger.set_run_metrics(
            {
                "network_parameter_counts": profile_param_counts,
            }
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
