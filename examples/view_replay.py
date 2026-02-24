"""Replay viewer for saved training replay JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlrlgym import EnvConfig, PettingZooParallelRLRLGym
from rlrlgym.models import AgentState, EnvState


def _state_from_payload(frame: dict) -> EnvState:
    tile_interactions = {
        (int(row["position"][0]), int(row["position"][1])): int(row["count"])
        for row in frame.get("tile_interactions", [])
    }
    ground_items = {
        (int(row["position"][0]), int(row["position"][1])): list(row.get("items", []))
        for row in frame.get("ground_items", [])
    }
    agents = {}
    for aid, row in frame.get("agents", {}).items():
        agent = AgentState(
            agent_id=str(row["agent_id"]),
            position=(int(row["position"][0]), int(row["position"][1])),
            profile_name=str(row.get("profile_name", "human")),
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
        )
        agents[aid] = agent
    return EnvState(
        grid=frame["grid"],
        tile_interactions=tile_interactions,
        ground_items=ground_items,
        agents=agents,
        step_count=int(frame.get("step_count", 0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="View a saved episode replay")
    parser.add_argument("replay_path", type=str, help="Path to *.replay.json file")
    parser.add_argument("--title", type=str, default="RLRLGym Replay Viewer")
    args = parser.parse_args()

    replay_path = Path(args.replay_path)
    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    frames_raw = payload.get("frames", [])
    actions_raw = payload.get("actions", [])
    if not frames_raw:
        raise ValueError(f"No frames in replay file: {replay_path}")

    states = [_state_from_payload(frame) for frame in frames_raw]
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
    action_log = [
        {str(aid): int(action) for aid, action in dict(row).items()}
        for row in actions_raw
        if isinstance(row, dict)
    ]
    env.play_frames_in_window(states, title=args.title, playback_actions=action_log)
    env.run_render_window()


if __name__ == "__main__":
    main()
