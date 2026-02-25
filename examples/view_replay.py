"""Replay viewer for saved training replay JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlrlgym import EnvConfig, PettingZooParallelRLRLGym
from rlrlgym.models import AgentState, ChestState, EnvState, MonsterState


def _state_from_payload(frame: dict) -> EnvState:
    tile_interactions = {
        (int(row["position"][0]), int(row["position"][1])): int(row["count"])
        for row in frame.get("tile_interactions", [])
    }
    ground_items = {
        (int(row["position"][0]), int(row["position"][1])): list(row.get("items", []))
        for row in frame.get("ground_items", [])
    }
    chests = {
        (int(row["position"][0]), int(row["position"][1])): ChestState(
            position=(int(row["position"][0]), int(row["position"][1])),
            opened=bool(row.get("opened", False)),
            locked=bool(row.get("locked", False)),
            loot=list(row.get("loot", [])),
        )
        for row in frame.get("chests", [])
    }
    agents = {}
    for aid, row in frame.get("agents", {}).items():
        agent = AgentState(
            agent_id=str(row["agent_id"]),
            position=(int(row["position"][0]), int(row["position"][1])),
            profile_name=str(row.get("profile_name", "human")),
            race_name=str(row.get("race_name", row.get("profile_name", "human"))),
            class_name=str(row.get("class_name", "wanderer")),
            hp=int(row.get("hp", 0)),
            max_hp=int(row.get("max_hp", 0)),
            hunger=int(row.get("hunger", 0)),
            max_hunger=int(row.get("max_hunger", 0)),
            inventory=list(row.get("inventory", [])),
            equipped=list(row.get("equipped", [])),
            alive=bool(row.get("alive", True)),
            visited={
                (int(pos[0]), int(pos[1]))
                for pos in row.get("visited", [])
            },
            wait_streak=int(row.get("wait_streak", 0)),
            recent_positions=[
                (int(pos[0]), int(pos[1]))
                for pos in row.get("recent_positions", [])
            ],
            strength=int(row.get("strength", 5)),
            dexterity=int(row.get("dexterity", 5)),
            intellect=int(row.get("intellect", 5)),
            skills={str(k): int(v) for k, v in dict(row.get("skills", {})).items()},
            skill_xp={str(k): int(v) for k, v in dict(row.get("skill_xp", {})).items()},
        )
        agents[aid] = agent
    monsters = {}
    for row in frame.get("monsters", []):
        entity_id = str(row.get("entity_id", "monster"))
        monster = MonsterState(
            entity_id=entity_id,
            monster_id=str(row.get("monster_id", entity_id)),
            name=str(row.get("name", row.get("monster_id", entity_id))),
            symbol=str(row.get("symbol", "M"))[:1] or "M",
            color=str(row.get("color", "red")),
            position=(int(row["position"][0]), int(row["position"][1])),
            hp=int(row.get("hp", 1)),
            max_hp=int(row.get("max_hp", 1)),
            acc=int(row.get("acc", 0)),
            eva=int(row.get("eva", 0)),
            dmg_min=int(row.get("dmg_min", 1)),
            dmg_max=int(row.get("dmg_max", 1)),
            dr_min=int(row.get("dr_min", 0)),
            dr_max=int(row.get("dr_max", 0)),
            alive=bool(row.get("alive", True)),
        )
        monsters[entity_id] = monster
    return EnvState(
        grid=frame["grid"],
        tile_interactions=tile_interactions,
        ground_items=ground_items,
        agents=agents,
        chests=chests,
        monsters=monsters,
        step_count=int(frame.get("step_count", 0)),
    )


def _fallback_step_logs(
    states: list[EnvState], actions_raw: list[dict]
) -> list[dict]:
    out: list[dict] = []
    for i in range(1, len(states)):
        prev = states[i - 1]
        curr = states[i]
        acts = actions_raw[i - 1] if (i - 1) < len(actions_raw) else {}
        agents = {}
        for aid, agent in curr.agents.items():
            prev_agent = prev.agents.get(aid)
            events = []
            if prev_agent is not None and prev_agent.hp > agent.hp:
                events.append("damage_taken")
            if prev_agent is not None and prev_agent.alive and not agent.alive:
                events.append("death")
            agents[aid] = {
                "action": int(acts.get(aid, -1)) if isinstance(acts, dict) else -1,
                "reward": 0.0,
                "events": events,
                "terminated": bool(not agent.alive),
                "truncated": False,
                "death_reason": "unknown" if "death" in events else None,
                "winner": False,
            }
        agent_damage = []
        for aid, agent in curr.agents.items():
            prev_agent = prev.agents.get(aid)
            if prev_agent is None:
                continue
            dmg = int(prev_agent.hp) - int(agent.hp)
            if dmg > 0:
                agent_damage.append(
                    {"agent_id": aid, "amount": dmg, "source": "unknown"}
                )
        monster_deaths = []
        monster_damage = []
        for entity_id, mon in curr.monsters.items():
            prev_mon = prev.monsters.get(entity_id)
            if prev_mon is None:
                continue
            dmg = int(prev_mon.hp) - int(mon.hp)
            if dmg > 0:
                monster_damage.append(
                    {
                        "entity_id": entity_id,
                        "monster_id": mon.monster_id,
                        "amount": dmg,
                        "hp_before": int(prev_mon.hp),
                        "hp_after": int(mon.hp),
                        "hp_max": int(mon.max_hp),
                        "source": "unknown",
                    }
                )
            if prev_mon.alive and not mon.alive:
                monster_deaths.append(
                    {
                        "entity_id": entity_id,
                        "monster_id": mon.monster_id,
                        "reason": "unknown",
                    }
                )
        out.append(
            {
                "agents": agents,
                "agent_damage": agent_damage,
                "monster_damage": monster_damage,
                "monster_deaths": monster_deaths,
            }
        )
    return out


def _load_replay_payload(replay_path: Path) -> tuple[list[EnvState], list[dict]]:
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    frames_raw = payload.get("frames", [])
    actions_raw = payload.get("actions", [])
    step_logs_raw = payload.get("step_logs", [])
    if not frames_raw:
        raise ValueError(f"No frames in replay file: {replay_path}")

    states = [_state_from_payload(frame) for frame in frames_raw]
    action_log: list[dict] = []
    if isinstance(step_logs_raw, list) and step_logs_raw:
        for row in step_logs_raw:
            if isinstance(row, dict):
                action_log.append(dict(row))
    else:
        raw_actions = [
            {str(aid): int(action) for aid, action in dict(row).items()}
            for row in actions_raw
            if isinstance(row, dict)
        ]
        action_log = _fallback_step_logs(states, raw_actions)
    return states, action_log


def main() -> None:
    parser = argparse.ArgumentParser(description="View a saved episode replay")
    parser.add_argument("replay_path", type=str, help="Path to *.replay.json file")
    parser.add_argument("--title", type=str, default="RLRLGym Replay Viewer")
    args = parser.parse_args()

    replay_path = Path(args.replay_path).resolve()
    states, action_log = _load_replay_payload(replay_path)
    first = states[0]
    width = len(first.grid[0]) if first.grid else 1
    height = len(first.grid)
    n_agents = max(1, len(first.agents))

    env = PettingZooParallelRLRLGym(
        EnvConfig(
            width=width,
            height=height,
            n_agents=n_agents,
            render_enabled=True,
        )
    )

    replay_files = sorted(replay_path.parent.glob("*.replay.json"))
    if replay_path in replay_files:
        current_idx = replay_files.index(replay_path)
    else:
        replay_files.append(replay_path)
        replay_files = sorted(replay_files)
        current_idx = replay_files.index(replay_path)

    def _load_idx(idx: int) -> None:
        nonlocal current_idx, states, action_log
        if idx < 0 or idx >= len(replay_files):
            return
        current_idx = idx
        p = replay_files[current_idx]
        states, action_log = _load_replay_payload(p)
        env.play_frames_in_window(
            states,
            title=f"{args.title} - {p.name}",
            playback_actions=action_log,
            on_prev_episode=_prev,
            on_next_episode=_next,
        )

    def _prev() -> None:
        _load_idx(current_idx - 1)

    def _next() -> None:
        _load_idx(current_idx + 1)

    env.play_frames_in_window(
        states,
        title=f"{args.title} - {replay_path.name}",
        playback_actions=action_log,
        on_prev_episode=_prev,
        on_next_episode=_next,
    )
    env.run_render_window()


if __name__ == "__main__":
    main()
