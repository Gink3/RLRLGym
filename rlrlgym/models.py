"""Data models for environment state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple


@dataclass
class TileDef:
    tile_id: str
    glyph: str
    color: str
    walkable: bool
    spawn_weight: float
    max_interactions: int
    loot_table: List[str]


@dataclass
class AgentState:
    agent_id: str
    position: Tuple[int, int]
    hp: int = 10
    hunger: int = 20
    inventory: List[str] = field(default_factory=list)
    equipped: List[str] = field(default_factory=list)
    alive: bool = True
    visited: Set[Tuple[int, int]] = field(default_factory=set)
    wait_streak: int = 0
    recent_positions: List[Tuple[int, int]] = field(default_factory=list)


@dataclass
class EnvState:
    grid: List[List[str]]
    tile_interactions: Dict[Tuple[int, int], int]
    ground_items: Dict[Tuple[int, int], List[str]]
    agents: Dict[str, AgentState]
    step_count: int = 0
