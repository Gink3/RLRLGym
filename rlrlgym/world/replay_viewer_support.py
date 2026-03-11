"""Pure helpers shared by replay viewer tests and UI code."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from rlrlgym.content.animals import load_animals
from rlrlgym.content.profiles import AgentProfile, load_profiles
from rlrlgym.content.tiles import load_tileset


OPAQUE_TILE_IDS = {
    "wall",
    "indestructible_wall",
    "rock_wall",
    "stone_wall",
    "wood_wall",
    "tree",
    "clay_forge",
    "clay_furnace",
    "clay_smelter",
}
DEFAULT_VISION_RANGE = 20
_AGENT_TOKEN_CACHE: Dict[str, re.Pattern[str]] = {}


def supported_animal_sprite_ids(
    path: str | Path = "data/base/animals.json",
) -> Set[str]:
    return set(load_animals(path).keys())


def supported_construction_sprite_ids(
    path: str | Path = "data/base/construction_recipes.json",
) -> Set[str]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    recipes = raw.get("recipes", [])
    out: Set[str] = set()
    if not isinstance(recipes, list):
        return out
    for row in recipes:
        if not isinstance(row, dict):
            continue
        tile_id = str(row.get("build_tile_id", "")).strip()
        station_id = str(row.get("build_station_id", "")).strip()
        if tile_id:
            out.add(tile_id)
        if station_id:
            out.add(station_id)
    return out


def resource_node_sprite_id(node_id: str, drop_item: str) -> str:
    node = str(node_id).strip().lower()
    drop = str(drop_item).strip().lower()
    if "timber" in node or "wood" in node or "tree" in node:
        return "timber"
    if "berry" in node:
        return "berries"
    if "grain" in node:
        return "grain"
    if "herb" in node:
        return "herb"
    if drop.endswith("_ore"):
        return "ore"
    if drop in {"stone", "clay", "flint"}:
        return drop
    return "generic"


def supported_resource_sprite_ids() -> Set[str]:
    return {"berries", "clay", "flint", "generic", "grain", "herb", "ore", "stone", "timber"}


def load_profile_map(
    path: str | Path = "data/base/agent/agent_profiles.json",
) -> Dict[str, AgentProfile]:
    return load_profiles(path)


def _agent_pattern(agent_id: str) -> re.Pattern[str]:
    pattern = _AGENT_TOKEN_CACHE.get(agent_id)
    if pattern is None:
        pattern = re.compile(rf"(?<![A-Za-z0-9_]){re.escape(agent_id)}(?![A-Za-z0-9_])")
        _AGENT_TOKEN_CACHE[agent_id] = pattern
    return pattern


def event_involves_agent(event: object, agent_id: str) -> bool:
    return bool(_agent_pattern(str(agent_id)).search(str(event)))


def _selected_agent_match(events: Sequence[object], selected_agents: Set[str]) -> bool:
    if not selected_agents:
        return True
    return any(event_involves_agent(evt, aid) for aid in selected_agents for evt in events)


def build_activity_log_lines(step: dict, selected_agents: Set[str]) -> List[str]:
    lines: List[str] = []
    agents = step.get("agents", {})
    if isinstance(agents, dict):
        for aid, row in sorted(agents.items()):
            if not isinstance(row, dict):
                continue
            events = row.get("events", [])
            event_list = events if isinstance(events, list) else []
            actor_selected = not selected_agents or aid in selected_agents
            involved = actor_selected or _selected_agent_match(event_list, selected_agents)
            if not involved:
                continue
            action = row.get("action", -1)
            reward = float(row.get("reward", 0.0))
            status_bits = []
            if bool(row.get("terminated", False)):
                status_bits.append("terminated")
            if bool(row.get("truncated", False)):
                status_bits.append("truncated")
            death_reason = row.get("death_reason")
            if death_reason:
                status_bits.append(f"death={death_reason}")
            if bool(row.get("winner", False)):
                status_bits.append("winner")
            suffix = f" [{' '.join(status_bits)}]" if status_bits else ""
            lines.append(f"{aid}: action={action} reward={reward:.3f}{suffix}")
            for evt in event_list:
                lines.append(f"  - {evt}")
    for row in step.get("agent_damage", []) or []:
        if not isinstance(row, dict):
            continue
        aid = str(row.get("agent_id", "")).strip()
        source = str(row.get("source", "")).strip()
        if selected_agents and aid not in selected_agents and not any(
            event_involves_agent(source, sel) for sel in selected_agents
        ):
            continue
        lines.append(f"damage: {aid} <- {source or 'unknown'} ({int(row.get('amount', 0))})")
    for row in step.get("monster_damage", []) or []:
        if not isinstance(row, dict):
            continue
        source = str(row.get("source", "")).strip()
        if selected_agents and not any(event_involves_agent(source, sel) for sel in selected_agents):
            continue
        monster_id = str(row.get("monster_id", row.get("entity_id", "monster"))).strip()
        lines.append(f"monster_damage: {monster_id} <- {source or 'unknown'} ({int(row.get('amount', 0))})")
    for row in step.get("monster_deaths", []) or []:
        if not isinstance(row, dict):
            continue
        reason = str(row.get("reason", "")).strip()
        if selected_agents and not any(event_involves_agent(reason, sel) for sel in selected_agents):
            continue
        monster_id = str(row.get("monster_id", row.get("entity_id", "monster"))).strip()
        lines.append(f"monster_death: {monster_id} ({reason or 'unknown'})")
    return lines


def _tile_blocks_los(tile_id: str, tile_defs: Dict[str, object]) -> bool:
    tile = str(tile_id)
    if tile in OPAQUE_TILE_IDS:
        return True
    td = tile_defs.get(tile)
    if td is None:
        return False
    return not bool(getattr(td, "walkable", True))


def _line_points(a: Tuple[int, int], b: Tuple[int, int]) -> List[Tuple[int, int]]:
    x0, y0 = int(a[0]), int(a[1])
    x1, y1 = int(b[0]), int(b[1])
    points: List[Tuple[int, int]] = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return points


def visible_tiles_for_agent(
    frame: dict,
    agent_id: str,
    tile_defs: Dict[str, object] | None = None,
    profiles: Dict[str, AgentProfile] | None = None,
    *,
    vision_range_default: int = DEFAULT_VISION_RANGE,
    enable_los: bool = True,
) -> Set[Tuple[int, int]]:
    grid = frame.get("grid", [])
    agents = frame.get("agents", {})
    if not isinstance(grid, list) or not grid or not isinstance(agents, dict):
        return set()
    agent = agents.get(agent_id)
    if not isinstance(agent, dict) or not bool(agent.get("alive", True)):
        return set()
    pos = agent.get("position", [])
    if not isinstance(pos, list) or len(pos) != 2:
        return set()
    r0 = int(pos[0])
    c0 = int(pos[1])
    tile_defs = tile_defs or load_tileset("data/base/tiles.json")
    profiles = profiles or load_profile_map()
    profile_name = str(agent.get("profile_name", "")).strip()
    profile = profiles.get(profile_name)
    skills = agent.get("skills", {})
    exploration_bonus = 0
    if isinstance(skills, dict):
        try:
            exploration_bonus = max(0, int(skills.get("exploration", 0)))
        except Exception:
            exploration_bonus = 0
    if profile is not None:
        view_h = max(1, max(int(profile.view_height), int(vision_range_default)) + exploration_bonus)
        view_w = max(1, max(int(profile.view_width), int(vision_range_default)) + exploration_bonus)
    else:
        view_h = max(1, int(vision_range_default) + exploration_bonus)
        view_w = max(1, int(vision_range_default) + exploration_bonus)
    start_r = r0 - (view_h // 2)
    start_c = c0 - (view_w // 2)
    out: Set[Tuple[int, int]] = set()
    for rr in range(start_r, start_r + view_h):
        for cc in range(start_c, start_c + view_w):
            if rr < 0 or cc < 0 or rr >= len(grid) or cc >= len(grid[0]):
                continue
            if not enable_los:
                out.add((rr, cc))
                continue
            blocked = False
            for lr, lc in _line_points((r0, c0), (rr, cc))[1:-1]:
                if lr < 0 or lc < 0 or lr >= len(grid) or lc >= len(grid[0]):
                    blocked = True
                    break
                if _tile_blocks_los(str(grid[lr][lc]), tile_defs):
                    blocked = True
                    break
            if not blocked:
                out.add((rr, cc))
    return out


def collect_agent_ids(frame: dict, step_logs: Iterable[dict]) -> List[str]:
    out: Set[str] = set()
    agents = frame.get("agents", {})
    if isinstance(agents, dict):
        out.update(str(aid) for aid in agents.keys())
    for row in step_logs:
        if not isinstance(row, dict):
            continue
        step_agents = row.get("agents", {})
        if isinstance(step_agents, dict):
            out.update(str(aid) for aid in step_agents.keys())
    return sorted(out)
