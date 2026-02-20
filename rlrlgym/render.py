"""ASCII renderer, playback controls, and optional windowed viewer."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

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


class RenderWindow:
    """Tkinter-based viewer for rendering states in a dedicated window."""

    def __init__(self, tiles: Dict[str, TileDef], title: str = "RLRLGym Viewer") -> None:
        try:
            import tkinter as tk
            from tkinter import ttk
        except Exception as exc:  # pragma: no cover - depends on host GUI support
            raise RuntimeError("Tkinter is required for window rendering") from exc

        self._tk = tk
        self._ttk = ttk
        self.renderer = AsciiRenderer(tiles)
        self.frames: List[str] = []
        self.cursor = 0
        self.speed = 1.0
        self.paused = True
        self._playing = False
        self._state: Optional[EnvState] = None
        self._focus_choices: List[str] = ["all"]

        self.root = tk.Tk()
        self.root.title(title)

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=6, pady=6)
        ttk.Button(controls, text="Play", command=self.play).pack(side="left")
        ttk.Button(controls, text="Pause", command=self.pause).pack(side="left", padx=(4, 0))
        ttk.Button(controls, text="Step", command=self.step).pack(side="left", padx=(4, 0))
        ttk.Button(controls, text="Fast x2", command=self.fast_forward).pack(side="left", padx=(4, 0))

        ttk.Label(controls, text="Focus").pack(side="left", padx=(10, 2))
        self.focus_var = tk.StringVar(value="all")
        self.focus_menu = ttk.Combobox(
            controls, textvariable=self.focus_var, values=self._focus_choices, width=10, state="readonly"
        )
        self.focus_menu.pack(side="left")
        self.focus_menu.bind("<<ComboboxSelected>>", self._on_focus_change)

        ttk.Label(controls, text="Zoom").pack(side="left", padx=(10, 2))
        self.zoom_var = tk.IntVar(value=0)
        self.zoom_scale = ttk.Scale(controls, from_=0, to=10, variable=self.zoom_var, command=self._on_zoom_change)
        self.zoom_scale.pack(side="left", fill="x", expand=True)

        self.text = tk.Text(self.root, width=100, height=35, font=("Courier", 11))
        self.text.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.text.configure(state="disabled")

    def _current_focus(self) -> Optional[str]:
        focus = self.focus_var.get().strip()
        return None if not focus or focus == "all" else focus

    def _render_from_state(self) -> None:
        if self._state is None:
            return
        frame = self.renderer.render(
            self._state,
            focus_agent=self._current_focus(),
            zoom=int(self.zoom_var.get()),
            color=False,
        )
        self._set_text(frame)

    def _on_focus_change(self, _evt=None) -> None:
        self._render_from_state()

    def _on_zoom_change(self, _evt=None) -> None:
        self._render_from_state()

    def _set_text(self, frame: str) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", self._tk.END)
        self.text.insert(self._tk.END, frame)
        self.text.configure(state="disabled")

    def set_frames(self, frames: List[str]) -> None:
        self.frames = list(frames)
        self.cursor = 0
        if self.frames:
            self._set_text(self.frames[0])

    def update_state(self, state: EnvState, focus_choices: Optional[List[str]] = None) -> None:
        self._state = state
        if focus_choices is not None and focus_choices:
            self._focus_choices = ["all"] + focus_choices
            self.focus_menu.configure(values=self._focus_choices)
            if self.focus_var.get() not in self._focus_choices:
                self.focus_var.set("all")
        self._render_from_state()
        self.pump()

    def play(self) -> None:
        self.paused = False
        if self.frames and not self._playing:
            self._playing = True
            self._tick()

    def pause(self) -> None:
        self.paused = True

    def fast_forward(self, factor: float = 2.0) -> None:
        self.speed = max(0.1, self.speed * factor)

    def step(self) -> None:
        if not self.frames:
            return
        self.cursor = min(len(self.frames) - 1, self.cursor + 1)
        self._set_text(self.frames[self.cursor])
        self.pump()

    def _tick(self) -> None:
        if not self._playing:
            return
        if self.paused:
            self.root.after(50, self._tick)
            return
        if not self.frames:
            self._playing = False
            return
        self._set_text(self.frames[self.cursor])
        if self.cursor < len(self.frames) - 1:
            self.cursor += 1
            delay_ms = int(1000.0 / self.speed)
            self.root.after(max(1, delay_ms), self._tick)
        else:
            self._playing = False

    def pump(self) -> None:
        self.root.update_idletasks()
        self.root.update()

    def close(self) -> None:
        self._playing = False
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
