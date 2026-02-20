"""ASCII renderer and playback controls."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from .models import EnvState, TileDef

ANSI = {
    "black": "\033[30m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
    "white": "\033[37m",
    "bright_black": "\033[90m",
    "reset": "\033[0m",
}


class AsciiRenderer:
    def __init__(self, tiles: Dict[str, TileDef]) -> None:
        self.tiles = tiles

    def render(
        self,
        state: EnvState,
        focus_agent: Optional[str] = None,
        zoom: int = 0,
        color: bool = True,
    ) -> str:
        min_r, max_r = 0, len(state.grid) - 1
        min_c, max_c = 0, len(state.grid[0]) - 1

        if focus_agent and focus_agent in state.agents and zoom > 0:
            ar, ac = state.agents[focus_agent].position
            min_r = max(0, ar - zoom)
            max_r = min(len(state.grid) - 1, ar + zoom)
            min_c = max(0, ac - zoom)
            max_c = min(len(state.grid[0]) - 1, ac + zoom)

        agent_positions = {a.position: aid for aid, a in state.agents.items() if a.alive}
        lines: List[str] = []
        for r in range(min_r, max_r + 1):
            row_chars: List[str] = []
            for c in range(min_c, max_c + 1):
                pos = (r, c)
                if pos in agent_positions:
                    symbol = agent_positions[pos][0].upper()
                    row_chars.append(f"\033[96m{symbol}\033[0m" if color else symbol)
                    continue
                tile = self.tiles[state.grid[r][c]]
                glyph = tile.glyph
                if color:
                    row_chars.append(f"{ANSI.get(tile.color, ANSI['white'])}{glyph}{ANSI['reset']}")
                else:
                    row_chars.append(glyph)
            lines.append("".join(row_chars))
        return "\n".join(lines)


@dataclass
class PlaybackController:
    frames: List[str]
    speed: float = 1.0
    paused: bool = False
    cursor: int = 0

    def pause(self) -> None:
        self.paused = True

    def play(self) -> None:
        self.paused = False

    def set_speed(self, speed: float) -> None:
        self.speed = max(0.1, speed)

    def fast_forward(self, factor: float = 2.0) -> None:
        self.speed = max(0.1, self.speed * factor)

    def step(self, steps: int = 1) -> str:
        self.cursor = min(len(self.frames) - 1, self.cursor + steps)
        return self.frames[self.cursor]

    def run(self, limit: Optional[int] = None) -> Iterable[str]:
        emitted = 0
        while self.cursor < len(self.frames):
            if self.paused:
                time.sleep(0.05)
                continue
            yield self.frames[self.cursor]
            emitted += 1
            if limit is not None and emitted >= limit:
                break
            self.cursor += 1
            time.sleep(1.0 / self.speed)
