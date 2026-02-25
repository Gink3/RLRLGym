"""Window-first renderer and playback controls."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .constants import ACTION_NAMES
from .models import EnvState, TileDef

TK_COLORS = {
    "black": "#000000",
    "red": "#e74c3c",
    "green": "#2ecc71",
    "yellow": "#f1c40f",
    "blue": "#3498db",
    "magenta": "#e056fd",
    "cyan": "#22a6b3",
    "white": "#ecf0f1",
    "bright_black": "#7f8c8d",
}

PROFILE_AGENT_STYLE = {
    "human": ("H", "#3498db"),
    "orc": ("O", "#2ecc71"),
}
DEFAULT_AGENT_STYLE = ("A", "#6dd5ff")


class AsciiRenderer:
    """Produces tile+color cells for window rendering."""

    def __init__(self, tiles: Dict[str, TileDef]) -> None:
        self.tiles = tiles

    def render_cells(
        self,
        state: EnvState,
        focus_agent: Optional[str] = None,
        zoom: int = 0,
    ) -> List[List[Tuple[str, str]]]:
        zoom = max(0, min(10, int(zoom)))
        min_r, max_r = 0, len(state.grid) - 1
        min_c, max_c = 0, len(state.grid[0]) - 1

        if focus_agent and focus_agent in state.agents and zoom > 0:
            ar, ac = state.agents[focus_agent].position
            min_r = max(0, ar - zoom)
            max_r = min(len(state.grid) - 1, ar + zoom)
            min_c = max(0, ac - zoom)
            max_c = min(len(state.grid[0]) - 1, ac + zoom)

        agent_positions = {
            a.position: aid for aid, a in state.agents.items() if a.alive
        }
        chest_positions = {
            pos
            for pos, chest in state.chests.items()
            if not chest.opened
        }
        monster_positions = {
            monster.position: monster
            for monster in state.monsters.values()
            if monster.alive
        }
        cells: List[List[Tuple[str, str]]] = []
        for r in range(min_r, max_r + 1):
            row_cells: List[Tuple[str, str]] = []
            for c in range(min_c, max_c + 1):
                pos = (r, c)
                if pos in agent_positions:
                    agent_id = agent_positions[pos]
                    agent = state.agents[agent_id]
                    symbol, color = PROFILE_AGENT_STYLE.get(
                        agent.profile_name, DEFAULT_AGENT_STYLE
                    )
                    row_cells.append((symbol, color))
                    continue
                if pos in monster_positions:
                    monster = monster_positions[pos]
                    row_cells.append((monster.symbol, TK_COLORS.get(monster.color, "#ff7675")))
                    continue
                if pos in chest_positions:
                    row_cells.append(("C", TK_COLORS["yellow"]))
                    continue
                tile = self.tiles[state.grid[r][c]]
                row_cells.append(
                    (tile.glyph, TK_COLORS.get(tile.color, TK_COLORS["white"]))
                )
            cells.append(row_cells)
        return cells


@dataclass
class PlaybackController:
    speed: float = 1.0
    paused: bool = False

    def pause(self) -> None:
        self.paused = True

    def play(self) -> None:
        self.paused = False

    def set_speed_1x(self) -> None:
        self.speed = 1.0

    def set_speed_2x(self) -> None:
        self.speed = 2.0

    def set_speed_5x(self) -> None:
        self.speed = 5.0


class RenderWindow:
    """Tkinter-based viewer for rendering states in a dedicated window."""

    def __init__(
        self, tiles: Dict[str, TileDef], title: str = "RLRLGym Viewer"
    ) -> None:
        try:
            import tkinter as tk
            from tkinter import ttk
        except Exception as exc:  # pragma: no cover - depends on host GUI support
            raise RuntimeError("Tkinter is required for window rendering") from exc

        self._tk = tk
        self._ttk = ttk
        self.renderer = AsciiRenderer(tiles)
        self.playback = PlaybackController()

        self._live_state: Optional[EnvState] = None
        self._playback_states: List[EnvState] = []
        self._playback_actions: List[Dict[str, int]] = []
        self._cursor = 0
        self._playing = False
        self._focus_choices: List[str] = ["all"]

        self.root = tk.Tk()
        self.root.title(title)

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=6, pady=6)
        ttk.Button(controls, text="Play", command=self.play).pack(side="left")
        ttk.Button(controls, text="Pause", command=self.pause).pack(
            side="left", padx=(4, 0)
        )
        ttk.Button(controls, text="Step", command=self.step).pack(
            side="left", padx=(4, 0)
        )
        ttk.Button(controls, text="1x", command=self.playback.set_speed_1x).pack(
            side="left", padx=(8, 0)
        )
        ttk.Button(controls, text="2x", command=self.playback.set_speed_2x).pack(
            side="left", padx=(2, 0)
        )
        ttk.Button(controls, text="5x", command=self.playback.set_speed_5x).pack(
            side="left", padx=(2, 0)
        )

        ttk.Label(controls, text="Focus").pack(side="left", padx=(10, 2))
        self.focus_var = tk.StringVar(value="all")
        self.focus_menu = ttk.Combobox(
            controls,
            textvariable=self.focus_var,
            values=self._focus_choices,
            width=10,
            state="readonly",
        )
        self.focus_menu.pack(side="left")
        self.focus_menu.bind("<<ComboboxSelected>>", self._on_focus_change)

        ttk.Label(controls, text="Zoom").pack(side="left", padx=(10, 2))
        self.zoom_var = tk.IntVar(value=0)
        self.zoom_scale = ttk.Scale(
            controls,
            from_=0,
            to=10,
            variable=self.zoom_var,
            command=self._on_zoom_change,
        )
        self.zoom_scale.pack(side="left", fill="x", expand=True)

        content = ttk.Frame(self.root)
        content.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self.text = tk.Text(
            content,
            width=100,
            height=35,
            font=("Courier", 11),
            bg="#1e1e1e",
            fg="#ecf0f1",
        )
        self.text.pack(side="left", fill="both", expand=True)
        self.text.configure(state="disabled")
        self.action_log_text = tk.Text(
            content,
            width=48,
            height=35,
            font=("Courier", 10),
            bg="#10161f",
            fg="#dbe7ff",
        )
        self.action_log_text.pack(side="left", fill="y", padx=(8, 0))
        self.action_log_text.configure(state="disabled")

        self._configure_color_tags()

    def _configure_color_tags(self) -> None:
        self.text.configure(state="normal")
        profile_colors = {style[1] for style in PROFILE_AGENT_STYLE.values()}
        for color in set(TK_COLORS.values()) | profile_colors | {
            DEFAULT_AGENT_STYLE[1]
        }:
            self.text.tag_configure(color, foreground=color)
        self.text.configure(state="disabled")

    def _current_focus(self) -> Optional[str]:
        focus = self.focus_var.get().strip()
        return None if not focus or focus == "all" else focus

    def _on_focus_change(self, _evt=None) -> None:
        self._redraw()

    def _on_zoom_change(self, _evt=None) -> None:
        z = int(self.zoom_var.get())
        self.zoom_var.set(max(0, min(10, z)))
        self._redraw()

    def _active_state(self) -> Optional[EnvState]:
        if self._playback_states:
            return self._playback_states[self._cursor]
        return self._live_state

    def _set_focus_choices(self, focus_choices: Optional[List[str]]) -> None:
        if focus_choices:
            self._focus_choices = ["all"] + focus_choices
            self.focus_menu.configure(values=self._focus_choices)
            if self.focus_var.get() not in self._focus_choices:
                self.focus_var.set("all")

    def _draw_cells(self, cells: List[List[Tuple[str, str]]]) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", self._tk.END)
        for row in cells:
            for glyph, color in row:
                self.text.insert(self._tk.END, glyph, color)
            self.text.insert(self._tk.END, "\n")
        self.text.configure(state="disabled")

    def _redraw(self) -> None:
        state = self._active_state()
        if state is None:
            return
        cells = self.renderer.render_cells(
            state,
            focus_agent=self._current_focus(),
            zoom=int(self.zoom_var.get()),
        )
        self._draw_cells(cells)
        self._redraw_action_log()

    def _redraw_action_log(self) -> None:
        self.action_log_text.configure(state="normal")
        self.action_log_text.delete("1.0", self._tk.END)
        self.action_log_text.insert(self._tk.END, "Recent Actions\n")
        self.action_log_text.insert(self._tk.END, "--------------\n")

        if not self._playback_states:
            self.action_log_text.insert(self._tk.END, "No playback loaded.\n")
        elif not self._playback_actions or self._cursor == 0:
            self.action_log_text.insert(self._tk.END, "No actions yet.\n")
        else:
            total = min(len(self._playback_actions), self._cursor)
            shown = 0
            for step_idx in range(total, 0, -1):
                acts = self._playback_actions[step_idx - 1]
                step_num = step_idx
                rendered = ", ".join(
                    f"{aid}={ACTION_NAMES.get(int(a), str(a))}({int(a)})"
                    for aid, a in sorted(acts.items())
                )
                self.action_log_text.insert(
                    self._tk.END, f"step {step_num}: {rendered}\n"
                )
                shown += 1
                if shown >= 25:
                    break
        self.action_log_text.configure(state="disabled")

    def update_state(
        self, state: EnvState, focus_choices: Optional[List[str]] = None
    ) -> None:
        self._live_state = copy.deepcopy(state)
        self._set_focus_choices(focus_choices)
        if not self._playback_states:
            self._redraw()
        self.pump()

    def set_playback_states(
        self,
        states: List[EnvState],
        focus_choices: Optional[List[str]] = None,
        action_log: Optional[List[Dict[str, int]]] = None,
    ) -> None:
        self._playback_states = [copy.deepcopy(state) for state in states]
        self._playback_actions = [dict(x) for x in (action_log or [])]
        self._cursor = 0
        self._set_focus_choices(focus_choices)
        self._redraw()

    def clear_playback(self) -> None:
        self._playback_states = []
        self._playback_actions = []
        self._cursor = 0
        self._playing = False
        self._redraw()

    def play(self) -> None:
        self.playback.play()
        if self._playback_states and not self._playing:
            self._playing = True
            self._tick()

    def pause(self) -> None:
        self.playback.pause()

    def step(self) -> None:
        if not self._playback_states:
            return
        self._cursor = min(len(self._playback_states) - 1, self._cursor + 1)
        self._redraw()
        self.pump()

    def _tick(self) -> None:
        if not self._playing:
            return
        if self.playback.paused:
            self.root.after(50, self._tick)
            return
        if not self._playback_states:
            self._playing = False
            return

        self._redraw()
        if self._cursor < len(self._playback_states) - 1:
            self._cursor += 1
            delay_ms = int(1000.0 / self.playback.speed)
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
