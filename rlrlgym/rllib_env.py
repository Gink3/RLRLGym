"""RLlib adapter environment for RLRLGym."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .env import EnvConfig, PettingZooParallelRLRLGym
from .featurize import observation_vector_size, vectorize_observation


class RLRLGymRLlibEnv(MultiAgentEnv):
    """A minimal RLlib-compatible multi-agent env wrapper."""

    def __init__(self, config: Dict[str, object] | None = None) -> None:
        super().__init__()
        cfg = config or {}
        profile_map = cfg.get("agent_profile_map", {"agent_0": "human", "agent_1": "orc"})
        env_cfg = EnvConfig.from_json(str(cfg.get("env_config_path", "data/env_config.json")))
        env_cfg.width = int(cfg.get("width", env_cfg.width))
        env_cfg.height = int(cfg.get("height", env_cfg.height))
        env_cfg.max_steps = int(cfg.get("max_steps", env_cfg.max_steps))
        env_cfg.n_agents = int(cfg.get("n_agents", env_cfg.n_agents))
        env_cfg.render_enabled = bool(cfg.get("render_enabled", False))
        env_cfg.agent_profile_map = dict(profile_map)
        self.base = PettingZooParallelRLRLGym(
            env_cfg
        )
        self._obs_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(observation_vector_size(),),
            dtype=np.float32,
        )
        self._action_space = spaces.Discrete(12)
        self.possible_agents = list(self.base.possible_agents)
        self.agents = []
        self.observation_spaces = {aid: self._obs_space for aid in self.possible_agents}
        self.action_spaces = {aid: self._action_space for aid in self.possible_agents}
        self._done_agents: set[str] = set()
        self._episode_counter = 0
        self._replay_save_every = int(cfg.get("replay_save_every", 1000))
        out_dir = cfg.get("replay_output_dir", "")
        self._replay_output_dir = Path(out_dir) if out_dir else None
        self._save_latest_replay = bool(cfg.get("save_latest_replay", True))
        self._capture_replay = False
        self._replay_states = []
        self._replay_actions = []
        self._replay_step_logs = []
        self._curriculum_phases = self._parse_curriculum_phases(cfg.get("curriculum_phases"))

    @property
    def observation_space(self):
        # RLlib MultiAgentEnv expects a per-agent mapping on the new API stack.
        return self.observation_spaces

    @property
    def action_space(self):
        # RLlib MultiAgentEnv expects a per-agent mapping on the new API stack.
        return self.action_spaces

    def get_observation_space(self, agent_id):
        return self.observation_spaces[agent_id]

    def get_action_space(self, agent_id):
        return self.action_spaces[agent_id]

    def reset(self, *, seed=None, options=None):
        self._episode_counter += 1
        self._apply_curriculum_for_episode(self._episode_counter)
        obs, info = self.base.reset(seed=seed, options=options)
        self._done_agents = set()
        self.agents = list(self.base.agents)
        self._capture_replay = (
            self._replay_output_dir is not None
            and self._replay_save_every > 0
            and self._episode_counter % self._replay_save_every == 0
        )
        should_capture_any = self._capture_replay or (
            self._replay_output_dir is not None and self._save_latest_replay
        )
        self._replay_states = [self.base.capture_playback_state()] if should_capture_any else []
        self._replay_actions = []
        self._replay_step_logs = []
        vec_obs = {
            aid: np.array(vectorize_observation(o), dtype=np.float32)
            for aid, o in obs.items()
        }
        return vec_obs, info

    def _parse_curriculum_phases(self, raw) -> list[Dict[str, object]]:
        if not isinstance(raw, list):
            return []
        out: list[Dict[str, object]] = []
        for row in raw:
            if not isinstance(row, dict):
                continue
            out.append(dict(row))
        return out

    def _apply_curriculum_for_episode(self, episode_num: int) -> None:
        if not self._curriculum_phases:
            return
        selected: Dict[str, object] | None = None
        for phase in self._curriculum_phases:
            until = int(phase.get("until_episode", 0) or 0)
            if until > 0 and episode_num <= until:
                selected = phase
                break
        if selected is None:
            selected = self._curriculum_phases[-1]

        if "width" in selected:
            self.base.config.width = int(selected["width"])
        if "height" in selected:
            self.base.config.height = int(selected["height"])
        if "max_steps" in selected:
            self.base.config.max_steps = int(selected["max_steps"])
        if "monster_density" in selected:
            self.base.mapgen_cfg.monster_density = float(selected["monster_density"])
        if "chest_density" in selected:
            self.base.mapgen_cfg.chest_density = float(selected["chest_density"])
        if "combat_training_mode" in selected:
            self.base.config.combat_training_mode = bool(selected["combat_training_mode"])
        if "hunger_tick_enabled" in selected:
            self.base.config.hunger_tick_enabled = bool(selected["hunger_tick_enabled"])
        if "missed_attack_opportunity_penalty" in selected:
            self.base.config.missed_attack_opportunity_penalty = float(
                selected["missed_attack_opportunity_penalty"]
            )

    def step(self, action_dict):
        obs, rewards, terminations, truncations, info = self.base.step(action_dict)
        if self._replay_states:
            self._replay_states.append(self.base.capture_playback_state())
            self._replay_actions.append(
                {aid: int(a) for aid, a in dict(action_dict).items()}
            )
            prev_state = self._replay_states[-2] if len(self._replay_states) >= 2 else None
            curr_state = self._replay_states[-1]
            self._replay_step_logs.append(
                self._build_replay_step_log(
                    actions=dict(action_dict),
                    rewards=rewards,
                    terminations=terminations,
                    truncations=truncations,
                    info=info,
                    prev_state=prev_state,
                    curr_state=curr_state,
                )
            )
        vec_obs = {}
        out_rewards = {}
        out_infos = {}
        terminateds = {}
        truncateds = {}

        # RLlib must not receive repeated terminal signals for already-done agents.
        for aid in self.possible_agents:
            if aid in self._done_agents:
                continue

            # RLlib expects a final observation for any agent that just got
            # terminated/truncated in this step.
            if aid in obs:
                final_obs = obs[aid]
            elif aid in self.base.state.agents:
                final_obs = self.base._build_observation(aid)  # type: ignore[attr-defined]
            else:
                final_obs = None
            if final_obs is not None:
                vec_obs[aid] = np.array(vectorize_observation(final_obs), dtype=np.float32)
            if aid in rewards:
                out_rewards[aid] = rewards[aid]
            if aid in info:
                out_infos[aid] = info[aid]

            term = bool(terminations.get(aid, False))
            trunc = bool(truncations.get(aid, False))
            if term:
                terminateds[aid] = True
                self._done_agents.add(aid)
            if trunc:
                truncateds[aid] = True
                self._done_agents.add(aid)

        self.agents = [aid for aid in self.base.agents if aid not in self._done_agents]
        all_done = len(self._done_agents) == len(self.possible_agents)
        terminateds["__all__"] = all_done
        truncateds["__all__"] = all_done
        if all_done:
            if self._capture_replay:
                self._write_replay(
                    episode=self._episode_counter,
                    frames=self._replay_states,
                    action_history=self._replay_actions,
                    step_logs=self._replay_step_logs,
                )
            if self._replay_output_dir is not None and self._save_latest_replay:
                self._write_latest_replay(
                    episode=self._episode_counter,
                    frames=self._replay_states,
                    action_history=self._replay_actions,
                    step_logs=self._replay_step_logs,
                )
            self._capture_replay = False
            self._replay_states = []
            self._replay_actions = []
            self._replay_step_logs = []
        return vec_obs, out_rewards, terminateds, truncateds, out_infos

    def _write_replay(
        self,
        episode: int,
        frames: list,
        action_history: list[Dict[str, int]],
        step_logs: list[Dict[str, object]] | None = None,
    ) -> None:
        if self._replay_output_dir is None:
            return
        out = self._replay_output_dir / "replays"
        out.mkdir(parents=True, exist_ok=True)
        p = out / f"episode_{episode:06d}.replay.json"
        payload = {
            "schema_version": 1,
            "episode": int(episode),
            "frame_count": len(frames),
            "frames": [self._serialize_state(s) for s in frames],
            "actions": [{aid: int(a) for aid, a in x.items()} for x in action_history],
            "step_logs": list(step_logs or []),
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _write_latest_replay(
        self,
        episode: int,
        frames: list,
        action_history: list[Dict[str, int]],
        step_logs: list[Dict[str, object]] | None = None,
    ) -> None:
        if self._replay_output_dir is None:
            return
        out = self._replay_output_dir / "replays"
        out.mkdir(parents=True, exist_ok=True)
        p = out / "latest_episode.replay.json"
        payload = {
            "schema_version": 1,
            "episode": int(episode),
            "frame_count": len(frames),
            "frames": [self._serialize_state(s) for s in frames],
            "actions": [{aid: int(a) for aid, a in x.items()} for x in action_history],
            "step_logs": list(step_logs or []),
        }
        p.write_text(json.dumps(payload, indent=2), encoding="utf-8")

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
        for aid in self.possible_agents:
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
                    "visited": [[r, c] for (r, c) in sorted(agent.visited)],
                    "wait_streak": agent.wait_streak,
                    "recent_positions": [[r, c] for (r, c) in agent.recent_positions],
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
