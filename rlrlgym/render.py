"""Window-first renderer and playback controls."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from .constants import ACTION_NAMES
from .models import EnvState, TileDef
from .themes import (
    get_theme,
    list_theme_names,
    load_selected_theme,
    save_selected_theme,
    theme_label,
)

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

    def view_bounds(
        self,
        state: EnvState,
        focus_agent: Optional[str] = None,
        zoom: int = 0,
    ) -> Tuple[int, int, int, int]:
        zoom = max(0, min(10, int(zoom)))
        min_r, max_r = 0, len(state.grid) - 1
        min_c, max_c = 0, len(state.grid[0]) - 1
        if focus_agent and focus_agent in state.agents and zoom > 0:
            ar, ac = state.agents[focus_agent].position
            min_r = max(0, ar - zoom)
            max_r = min(len(state.grid) - 1, ar + zoom)
            min_c = max(0, ac - zoom)
            max_c = min(len(state.grid[0]) - 1, ac + zoom)
        return min_r, max_r, min_c, max_c

    def render_cells(
        self,
        state: EnvState,
        focus_agent: Optional[str] = None,
        zoom: int = 0,
    ) -> List[List[Tuple[str, str, bool]]]:
        min_r, max_r, min_c, max_c = self.view_bounds(
            state=state, focus_agent=focus_agent, zoom=zoom
        )

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
        cells: List[List[Tuple[str, str, bool]]] = []
        for r in range(min_r, max_r + 1):
            row_cells: List[Tuple[str, str, bool]] = []
            for c in range(min_c, max_c + 1):
                pos = (r, c)
                if pos in agent_positions:
                    agent_id = agent_positions[pos]
                    agent = state.agents[agent_id]
                    symbol, color = PROFILE_AGENT_STYLE.get(
                        agent.profile_name, DEFAULT_AGENT_STYLE
                    )
                    row_cells.append((symbol, color, True))
                    continue
                if pos in monster_positions:
                    monster = monster_positions[pos]
                    row_cells.append(
                        (monster.symbol, TK_COLORS.get(monster.color, "#ff7675"), False)
                    )
                    continue
                if pos in chest_positions:
                    row_cells.append(("C", TK_COLORS["yellow"], False))
                    continue
                tile = self.tiles[state.grid[r][c]]
                row_cells.append(
                    (tile.glyph, TK_COLORS.get(tile.color, TK_COLORS["white"]), False)
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
        self._theme_name = load_selected_theme()
        self._theme = get_theme(self._theme_name)

        self._live_state: Optional[EnvState] = None
        self._playback_states: List[EnvState] = []
        self._playback_actions: List[Dict[str, object]] = []
        self._cursor = 0
        self._playing = False
        self._focus_choices: List[str] = ["all"]
        self._last_view_bounds: Optional[Tuple[int, int, int, int]] = None
        self._tooltip_window = None
        self._tooltip_label = None
        self._tooltip_cell: Optional[Tuple[int, int]] = None
        self._on_prev_episode: Optional[Callable[[], None]] = None
        self._on_next_episode: Optional[Callable[[], None]] = None
        self._tileset_config: Dict[str, object] = {}
        self._tileset_image = None
        self._tileset_tile_size = 16
        self._tileset_photo_cache: Dict[Tuple[str, int], object] = {}
        self._canvas_images: List[object] = []
        self._last_tile_pixel_size = 16
        self._tileset_available = False

        self.root = tk.Tk()
        self.root.title(title)

        controls = ttk.Frame(self.root)
        controls.pack(fill="x", padx=6, pady=6)
        ttk.Button(controls, text="Play", command=self.play).pack(side="left")
        ttk.Button(controls, text="Pause", command=self.pause).pack(
            side="left", padx=(4, 0)
        )
        ttk.Button(controls, text="Restart", command=self.restart).pack(
            side="left", padx=(4, 0)
        )
        ttk.Button(controls, text="Step", command=self.step).pack(
            side="left", padx=(4, 0)
        )
        ttk.Button(controls, text="Prev Ep", command=self.prev_episode).pack(
            side="left", padx=(10, 0)
        )
        ttk.Button(controls, text="Next Ep", command=self.next_episode).pack(
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
        self.highlight_agents_var = tk.BooleanVar(value=False)
        self.highlight_mode_var = tk.StringVar(value="background")
        ttk.Checkbutton(
            controls,
            text="Highlight Agents",
            variable=self.highlight_agents_var,
            command=self._on_highlight_toggle,
        ).pack(side="left", padx=(10, 2))
        self.highlight_mode_menu = ttk.Combobox(
            controls,
            textvariable=self.highlight_mode_var,
            values=["background", "outline"],
            width=11,
            state="readonly",
        )
        self.highlight_mode_menu.pack(side="left")
        self.highlight_mode_menu.bind("<<ComboboxSelected>>", self._on_highlight_mode_change)
        ttk.Label(controls, text="Render").pack(side="left", padx=(10, 2))
        self.render_mode_var = tk.StringVar(value="ascii")
        self.render_mode_menu = ttk.Combobox(
            controls,
            textvariable=self.render_mode_var,
            values=["ascii", "tileset"],
            width=9,
            state="readonly",
        )
        self.render_mode_menu.pack(side="left")
        self.render_mode_menu.bind("<<ComboboxSelected>>", self._on_render_mode_change)
        self.theme_var = tk.StringVar(value=self._theme_name)
        self.settings_button = ttk.Menubutton(controls, text="Settings")
        self.settings_menu = tk.Menu(self.settings_button, tearoff=0)
        self.theme_menu = tk.Menu(self.settings_menu, tearoff=0)
        for name in list_theme_names():
            self.theme_menu.add_radiobutton(
                label=theme_label(name),
                value=name,
                variable=self.theme_var,
                command=self._on_theme_change,
            )
        self.settings_menu.add_cascade(label="Theme", menu=self.theme_menu)
        self.settings_button["menu"] = self.settings_menu
        self.settings_button.pack(side="left", padx=(10, 0))

        content = ttk.Frame(self.root)
        content.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.panes = ttk.Panedwindow(content, orient="horizontal")
        self.panes.pack(fill="both", expand=True)
        map_frame = ttk.Frame(self.panes)
        log_frame = ttk.Frame(self.panes)
        self.panes.add(map_frame, weight=4)
        self.panes.add(log_frame, weight=2)
        self.right_panes = ttk.Panedwindow(log_frame, orient="vertical")
        self.right_panes.pack(fill="both", expand=True)
        action_frame = ttk.Frame(self.right_panes)
        stats_frame = ttk.Frame(self.right_panes)
        self.right_panes.add(action_frame, weight=3)
        self.right_panes.add(stats_frame, weight=2)
        self.ascii_map_frame = ttk.Frame(map_frame)
        self.tileset_map_frame = ttk.Frame(map_frame)
        self.ascii_map_frame.grid(row=0, column=0, sticky="nsew")
        self.tileset_map_frame.grid(row=0, column=0, sticky="nsew")

        self.text = tk.Text(
            self.ascii_map_frame,
            width=100,
            height=35,
            font=("Courier", 11),
            bg="#1e1e1e",
            fg="#ecf0f1",
            wrap="none",
        )
        self.text_vscroll = ttk.Scrollbar(
            self.ascii_map_frame, orient="vertical", command=self.text.yview
        )
        self.text_hscroll = ttk.Scrollbar(
            self.ascii_map_frame, orient="horizontal", command=self.text.xview
        )
        self.text.configure(
            yscrollcommand=self.text_vscroll.set, xscrollcommand=self.text_hscroll.set
        )
        self.text.grid(row=0, column=0, sticky="nsew")
        self.text_vscroll.grid(row=0, column=1, sticky="ns")
        self.text_hscroll.grid(row=1, column=0, sticky="ew")
        self.ascii_map_frame.rowconfigure(0, weight=1)
        self.ascii_map_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            self.tileset_map_frame,
            width=1000,
            height=700,
            bg="#0f1318",
            highlightthickness=0,
        )
        self.canvas_vscroll = ttk.Scrollbar(
            self.tileset_map_frame, orient="vertical", command=self.canvas.yview
        )
        self.canvas_hscroll = ttk.Scrollbar(
            self.tileset_map_frame, orient="horizontal", command=self.canvas.xview
        )
        self.canvas.configure(
            yscrollcommand=self.canvas_vscroll.set,
            xscrollcommand=self.canvas_hscroll.set,
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas_vscroll.grid(row=0, column=1, sticky="ns")
        self.canvas_hscroll.grid(row=1, column=0, sticky="ew")
        self.tileset_map_frame.rowconfigure(0, weight=1)
        self.tileset_map_frame.columnconfigure(0, weight=1)
        self.tileset_map_frame.lower()
        map_frame.rowconfigure(0, weight=1)
        map_frame.columnconfigure(0, weight=1)
        self.text.configure(state="disabled")
        self.text.bind("<Motion>", self._on_text_motion)
        self.text.bind("<Leave>", self._on_text_leave)
        self.canvas.bind("<Motion>", self._on_canvas_motion)
        self.canvas.bind("<Leave>", self._on_canvas_leave)
        self.action_log_text = tk.Text(
            action_frame,
            width=48,
            height=35,
            font=("Courier", 10),
            bg="#10161f",
            fg="#dbe7ff",
        )
        self.action_log_text.pack(side="left", fill="both", expand=True)
        self.action_log_text.configure(state="disabled")
        self.agent_stats_canvas = tk.Canvas(
            stats_frame,
            width=48,
            height=14,
            bg="#0e141c",
            highlightthickness=0,
        )
        self.agent_stats_vscroll = ttk.Scrollbar(
            stats_frame, orient="vertical", command=self.agent_stats_canvas.yview
        )
        self.agent_stats_canvas.configure(yscrollcommand=self.agent_stats_vscroll.set)
        self.agent_stats_canvas.pack(side="left", fill="both", expand=True)
        self.agent_stats_vscroll.pack(side="left", fill="y")
        self.agent_stats_canvas.bind("<Configure>", self._on_agent_stats_resize)
        self._base_map_font_size = 11
        self._base_log_font_size = 10
        self._base_stats_font_size = 10
        self._apply_zoom_font()

        self._configure_color_tags()
        self._tileset_available = self._load_tileset_assets()
        self._apply_theme()

    def _configure_color_tags(self) -> None:
        t = self._theme
        self.text.configure(state="normal")
        profile_colors = {style[1] for style in PROFILE_AGENT_STYLE.values()}
        for color in set(TK_COLORS.values()) | profile_colors | {
            DEFAULT_AGENT_STYLE[1]
        }:
            self.text.tag_configure(color, foreground=color)
            self.text.tag_configure(
                self._agent_bg_tag(color),
                background=t["accent"],
            )
        self.text.tag_configure("agent_outline", borderwidth=1, relief="solid")
        self.text.tag_configure("agent_bold")
        self.text.configure(state="disabled")

    def _current_focus(self) -> Optional[str]:
        focus = self.focus_var.get().strip()
        return None if not focus or focus == "all" else focus

    def _on_focus_change(self, _evt=None) -> None:
        self._redraw()

    def _on_zoom_change(self, _evt=None) -> None:
        z = int(self.zoom_var.get())
        self.zoom_var.set(max(0, min(10, z)))
        self._apply_zoom_font()
        self._redraw()

    def _on_render_mode_change(self, _evt=None) -> None:
        mode = self.render_mode_var.get()
        if mode == "tileset" and not self._tileset_available:
            self.render_mode_var.set("ascii")
        self._set_map_mode()
        self._redraw()

    def _on_theme_change(self) -> None:
        name = save_selected_theme(self.theme_var.get())
        self.theme_var.set(name)
        self._theme_name = name
        self._theme = get_theme(name)
        self._configure_color_tags()
        self._apply_theme()
        self._redraw()

    def _on_highlight_toggle(self) -> None:
        self._redraw()

    def _on_highlight_mode_change(self, _evt=None) -> None:
        if self.highlight_mode_var.get() not in {"background", "outline"}:
            self.highlight_mode_var.set("background")
        self._redraw()

    def _agent_bg_tag(self, color: str) -> str:
        return f"agent_bg_{color.strip('#')}"

    def _apply_theme(self) -> None:
        t = self._theme
        self.root.configure(bg=t["bg"])
        style = self._ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", background=t["panel"], foreground=t["text"])
        style.configure("TFrame", background=t["panel"])
        style.configure("TLabel", background=t["panel"], foreground=t["text"])
        style.configure(
            "TButton",
            background=t["panel_alt"],
            foreground=t["text"],
            bordercolor=t["border"],
        )
        style.map("TButton", background=[("active", t["primary_hover"])])
        style.configure(
            "TCombobox",
            fieldbackground=t["panel_alt"],
            background=t["panel_alt"],
            foreground=t["text"],
        )
        style.configure(
            "TCheckbutton",
            background=t["panel"],
            foreground=t["text"],
        )
        style.configure("TMenubutton", background=t["panel_alt"], foreground=t["text"])
        style.configure(
            "TScrollbar",
            background=t["panel_alt"],
            troughcolor=t["panel"],
        )
        self.text.configure(bg=t["bg"], fg=t["text"], insertbackground=t["text"])
        self.canvas.configure(bg=t["bg"])
        self.action_log_text.configure(
            bg=t["panel"],
            fg=t["text"],
            insertbackground=t["text"],
        )
        self.agent_stats_canvas.configure(bg=t["panel"])
        self.settings_menu.configure(
            background=t["panel"],
            foreground=t["text"],
            activebackground=t["primary"],
            activeforeground=t["text"],
            tearoff=False,
        )
        self.theme_menu.configure(
            background=t["panel"],
            foreground=t["text"],
            activebackground=t["primary"],
            activeforeground=t["text"],
            tearoff=False,
        )

    def _apply_zoom_font(self) -> None:
        zoom = int(self.zoom_var.get())
        # Zoom scales map glyph size for clearer replay inspection.
        font_size = max(8, min(26, self._base_map_font_size + zoom))
        self.text.configure(font=("Courier", font_size))
        self.text.tag_configure("agent_bold", font=("Courier", font_size, "bold"))
        # Keep side panes fixed-size for readability regardless of map zoom.
        self.action_log_text.configure(font=("Courier", self._base_log_font_size))
        # Agent stat cards are canvas-rendered with a fixed readable size.

    def _on_agent_stats_resize(self, _evt=None) -> None:
        self._redraw_agent_stats()

    def _set_map_mode(self) -> None:
        mode = self.render_mode_var.get()
        if mode == "tileset" and self._tileset_available:
            self.tileset_map_frame.lift()
        else:
            self.ascii_map_frame.lift()

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

    def _draw_cells(self, cells: List[List[Tuple[str, str, bool]]]) -> None:
        self.text.configure(state="normal")
        self.text.delete("1.0", self._tk.END)
        for row in cells:
            for glyph, color, is_agent in row:
                tags: Tuple[str, ...]
                if is_agent and bool(self.highlight_agents_var.get()):
                    if self.highlight_mode_var.get() == "outline":
                        tags = (color, "agent_outline", "agent_bold")
                    else:
                        tags = (color, self._agent_bg_tag(color), "agent_bold")
                else:
                    tags = (color,)
                self.text.insert(self._tk.END, glyph, tags)
            self.text.insert(self._tk.END, "\n")
        self.text.configure(state="disabled")

    def _load_tileset_assets(self) -> bool:
        cfg_path = Path("assets") / "tileset_basic.json"
        if not cfg_path.exists():
            return False
        try:
            from PIL import Image
        except Exception:
            return False
        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
            if not isinstance(cfg, dict):
                return False
            image_name = str(cfg.get("image", ""))
            if not image_name:
                return False
            image_path = cfg_path.parent / image_name
            if not image_path.exists():
                return False
            self._tileset_config = cfg
            self._tileset_image = Image.open(image_path).convert("RGBA")
            self._tileset_tile_size = int(cfg.get("tile_size", 16))
            return True
        except Exception:
            return False

    def _tileset_photo(self, tile_id: str, pixel_size: int):
        if not self._tileset_available:
            return None
        key = (tile_id, pixel_size)
        if key in self._tileset_photo_cache:
            return self._tileset_photo_cache[key]
        try:
            from PIL import ImageTk
        except Exception:
            return None
        assert isinstance(self._tileset_config, dict)
        tiles = self._tileset_config.get("tiles", {})
        if not isinstance(tiles, dict):
            return None
        spec = tiles.get(tile_id) or tiles.get("void")
        if not isinstance(spec, dict):
            return None
        col = int(spec.get("col", 0))
        row = int(spec.get("row", 0))
        s = int(self._tileset_tile_size)
        x0 = col * s
        y0 = row * s
        x1 = x0 + s
        y1 = y0 + s
        sub = self._tileset_image.crop((x0, y0, x1, y1))
        if pixel_size != s:
            sub = sub.resize((pixel_size, pixel_size), resample=0)
        photo = ImageTk.PhotoImage(sub)
        self._tileset_photo_cache[key] = photo
        return photo

    def _draw_tileset(self, state: EnvState) -> None:
        bounds = self._last_view_bounds
        if bounds is None:
            return
        min_r, max_r, min_c, max_c = bounds
        zoom = int(self.zoom_var.get())
        pixel_size = max(8, min(48, 16 + 2 * zoom))
        self._last_tile_pixel_size = pixel_size
        self.canvas.delete("all")
        self._canvas_images = []

        width_cells = max_c - min_c + 1
        height_cells = max_r - min_r + 1
        self.canvas.configure(
            scrollregion=(0, 0, width_cells * pixel_size, height_cells * pixel_size)
        )

        agent_positions = {a.position: aid for aid, a in state.agents.items() if a.alive}
        chest_positions = {
            pos for pos, chest in state.chests.items() if not chest.opened
        }
        monster_positions = {
            monster.position: monster for monster in state.monsters.values() if monster.alive
        }

        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                x = (c - min_c) * pixel_size
                y = (r - min_r) * pixel_size
                base_tile_id = state.grid[r][c]
                base_photo = self._tileset_photo(base_tile_id, pixel_size)
                if base_photo is not None:
                    self.canvas.create_image(x, y, anchor="nw", image=base_photo)
                    self._canvas_images.append(base_photo)

                pos = (r, c)
                if pos in chest_positions:
                    chest_photo = self._tileset_photo("chest", pixel_size)
                    if chest_photo is not None:
                        self.canvas.create_image(x, y, anchor="nw", image=chest_photo)
                        self._canvas_images.append(chest_photo)
                if pos in monster_positions:
                    mon = monster_positions[pos]
                    self.canvas.create_text(
                        x + (pixel_size // 2),
                        y + (pixel_size // 2),
                        text=str(mon.symbol),
                        fill=TK_COLORS.get(mon.color, "#ff7675"),
                        font=("Courier", max(8, pixel_size // 2), "bold"),
                    )
                if pos in agent_positions:
                    aid = agent_positions[pos]
                    agent = state.agents[aid]
                    symbol, color = PROFILE_AGENT_STYLE.get(
                        agent.profile_name, DEFAULT_AGENT_STYLE
                    )
                    self.canvas.create_oval(
                        x + 2,
                        y + 2,
                        x + pixel_size - 2,
                        y + pixel_size - 2,
                        fill="#111827",
                        outline=color,
                        width=2,
                    )
                    self.canvas.create_text(
                        x + (pixel_size // 2),
                        y + (pixel_size // 2),
                        text=symbol,
                        fill=color,
                        font=("Courier", max(8, pixel_size // 2), "bold"),
                    )

    def _capture_scroll_fraction(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        mode = self.render_mode_var.get()
        if mode == "tileset" and self._tileset_available:
            return self.canvas.xview(), self.canvas.yview()
        return self.text.xview(), self.text.yview()

    def _restore_scroll_fraction(
        self,
        xview: Tuple[float, float],
        yview: Tuple[float, float],
    ) -> None:
        mode = self.render_mode_var.get()
        x0 = float(xview[0]) if xview else 0.0
        y0 = float(yview[0]) if yview else 0.0
        x0 = max(0.0, min(1.0, x0))
        y0 = max(0.0, min(1.0, y0))
        if mode == "tileset" and self._tileset_available:
            self.canvas.xview_moveto(x0)
            self.canvas.yview_moveto(y0)
        else:
            self.text.xview_moveto(x0)
            self.text.yview_moveto(y0)

    def _redraw(self) -> None:
        state = self._active_state()
        if state is None:
            self._hide_tooltip()
            return
        prev_xview, prev_yview = self._capture_scroll_fraction()
        self._set_map_mode()
        self._last_view_bounds = self.renderer.view_bounds(
            state=state,
            focus_agent=self._current_focus(),
            zoom=int(self.zoom_var.get()),
        )
        if self.render_mode_var.get() == "tileset" and self._tileset_available:
            self._draw_tileset(state)
        else:
            cells = self.renderer.render_cells(
                state,
                focus_agent=self._current_focus(),
                zoom=int(self.zoom_var.get()),
            )
            self._draw_cells(cells)
        self._restore_scroll_fraction(prev_xview, prev_yview)
        self._redraw_action_log()
        self._redraw_agent_stats()

    def _on_text_leave(self, _evt=None) -> None:
        self._hide_tooltip()

    def _on_canvas_leave(self, _evt=None) -> None:
        self._hide_tooltip()

    def _on_text_motion(self, evt) -> None:
        state = self._active_state()
        bounds = self._last_view_bounds
        if state is None or bounds is None:
            self._hide_tooltip()
            return

        try:
            idx = self.text.index(f"@{evt.x},{evt.y}")
            line_str, col_str = idx.split(".")
            line = int(line_str)
            col = int(col_str)
        except Exception:
            self._hide_tooltip()
            return

        min_r, max_r, min_c, max_c = bounds
        height = max_r - min_r + 1
        width = max_c - min_c + 1
        if line < 1 or line > height or col < 0 or col >= width:
            self._hide_tooltip()
            return

        grid_r = min_r + (line - 1)
        grid_c = min_c + col
        if self._tooltip_cell == (grid_r, grid_c):
            return
        self._tooltip_cell = (grid_r, grid_c)
        self._show_tooltip(
            x=evt.x_root + 14,
            y=evt.y_root + 14,
            text=self._tile_tooltip_text(state, grid_r, grid_c),
        )

    def _on_canvas_motion(self, evt) -> None:
        state = self._active_state()
        bounds = self._last_view_bounds
        if state is None or bounds is None:
            self._hide_tooltip()
            return
        min_r, max_r, min_c, max_c = bounds
        width = max_c - min_c + 1
        height = max_r - min_r + 1
        s = max(1, int(self._last_tile_pixel_size))
        x = self.canvas.canvasx(evt.x)
        y = self.canvas.canvasy(evt.y)
        col = int(x // s)
        row = int(y // s)
        if row < 0 or col < 0 or row >= height or col >= width:
            self._hide_tooltip()
            return
        grid_r = min_r + row
        grid_c = min_c + col
        if self._tooltip_cell == (grid_r, grid_c):
            return
        self._tooltip_cell = (grid_r, grid_c)
        self._show_tooltip(
            x=evt.x_root + 14,
            y=evt.y_root + 14,
            text=self._tile_tooltip_text(state, grid_r, grid_c),
        )

    def _show_tooltip(self, x: int, y: int, text: str) -> None:
        t = self._theme
        if self._tooltip_window is None:
            win = self._tk.Toplevel(self.root)
            win.overrideredirect(True)
            win.attributes("-topmost", True)
            label = self._tk.Label(
                win,
                justify="left",
                bg=t["panel"],
                fg=t["text"],
                relief="solid",
                borderwidth=1,
                font=("Courier", 9),
                padx=6,
                pady=4,
            )
            label.pack()
            self._tooltip_window = win
            self._tooltip_label = label
        assert self._tooltip_window is not None
        assert self._tooltip_label is not None
        self._tooltip_label.configure(
            bg=t["panel"],
            fg=t["text"],
        )
        self._tooltip_label.configure(text=text)
        self._tooltip_window.geometry(f"+{int(x)}+{int(y)}")
        self._tooltip_window.deiconify()

    def _hide_tooltip(self) -> None:
        self._tooltip_cell = None
        if self._tooltip_window is not None:
            self._tooltip_window.withdraw()

    def _tile_tooltip_text(self, state: EnvState, r: int, c: int) -> str:
        tile_id = state.grid[r][c]
        tile = self.renderer.tiles[tile_id]
        lines = [
            f"pos=({r},{c})",
            f"tile={tile_id}",
            f"walkable={tile.walkable} interactions={tile.max_interactions}",
        ]
        if tile.loot_table:
            lines.append(f"tile_loot={','.join(tile.loot_table[:6])}")

        for aid, agent in sorted(state.agents.items()):
            if agent.alive and agent.position == (r, c):
                lines.append(
                    f"agent={aid} profile={agent.profile_name} hp={agent.hp}/{agent.max_hp} hunger={agent.hunger}/{agent.max_hunger}"
                )
                break

        for _, monster in sorted(state.monsters.items()):
            if monster.alive and monster.position == (r, c):
                lines.append(
                    f"monster={monster.name} ({monster.monster_id}) hp={monster.hp}/{monster.max_hp}"
                )
                break

        chest = state.chests.get((r, c))
        if chest is not None:
            lines.append(
                f"chest={'opened' if chest.opened else 'closed'} loot_count={len(chest.loot)}"
            )

        ground = state.ground_items.get((r, c), [])
        if ground:
            lines.append(f"ground_items={','.join(ground[:8])}")

        return "\n".join(lines)

    def _redraw_action_log(self) -> None:
        self.action_log_text.configure(state="normal")
        self.action_log_text.delete("1.0", self._tk.END)
        self.action_log_text.insert(self._tk.END, "Turn Log\n")
        self.action_log_text.insert(self._tk.END, "--------\n")

        if not self._playback_states:
            self.action_log_text.insert(self._tk.END, "No playback loaded.\n")
        elif not self._playback_actions or self._cursor == 0:
            self.action_log_text.insert(self._tk.END, "No turns yet.\n")
        else:
            total = min(len(self._playback_actions), self._cursor)
            shown = 0
            for step_idx in range(total, 0, -1):
                entry = self._playback_actions[step_idx - 1]
                step_num = step_idx
                self._write_action_log_entry(step_num, entry)
                shown += 1
                if shown >= 20:
                    break
        self.action_log_text.configure(state="disabled")

    def _write_action_log_entry(self, step_num: int, entry: Dict[str, object]) -> None:
        if "agents" in entry and isinstance(entry.get("agents"), dict):
            agents = entry["agents"]
            self.action_log_text.insert(self._tk.END, f"step {step_num}\n")
            for aid, row in sorted(agents.items()):
                if not isinstance(row, dict):
                    continue
                action = int(row.get("action", -1))
                action_name = ACTION_NAMES.get(action, str(action))
                reward = float(row.get("reward", 0.0))
                self.action_log_text.insert(
                    self._tk.END, f"  {aid}: {action_name}({action}) r={reward:+.3f}\n"
                )
                death_reason = row.get("death_reason")
                if death_reason:
                    self.action_log_text.insert(
                        self._tk.END, f"    DEATH: {str(death_reason)}\n"
                    )
                if bool(row.get("winner", False)):
                    self.action_log_text.insert(self._tk.END, "    WINNER\n")
            self.action_log_text.insert(self._tk.END, "\n")
            for dmg in entry.get("agent_damage", []):
                if not isinstance(dmg, dict):
                    continue
                self.action_log_text.insert(
                    self._tk.END,
                    f"  DAMAGE: {dmg.get('agent_id')} -{int(dmg.get('amount', 0))} ({dmg.get('source', 'unknown')})\n",
                )
            for mdmg in entry.get("monster_damage", []):
                if not isinstance(mdmg, dict):
                    continue
                self.action_log_text.insert(
                    self._tk.END,
                    (
                        f"  MONSTER_DAMAGE: {mdmg.get('monster_id')} [{mdmg.get('entity_id')}] "
                        f"-{int(mdmg.get('amount', 0))} "
                        f"({int(mdmg.get('hp_before', 0))}->{int(mdmg.get('hp_after', 0))}) "
                        f"hp={int(mdmg.get('hp_after', 0))}/{int(mdmg.get('hp_max', 0))} "
                        f"src={mdmg.get('source', 'unknown')}\n"
                    ),
                )
            for md in entry.get("monster_deaths", []):
                if not isinstance(md, dict):
                    continue
                self.action_log_text.insert(
                    self._tk.END,
                    f"  MONSTER_DEATH: {md.get('monster_id')} [{md.get('entity_id')}] {md.get('reason', 'unknown')}\n",
                )
            if entry.get("agent_damage") or entry.get("monster_damage") or entry.get("monster_deaths"):
                self.action_log_text.insert(self._tk.END, "\n")
            return

        acts = {
            str(aid): int(a)
            for aid, a in entry.items()
            if isinstance(a, (int, float))
        }
        rendered = ", ".join(
            f"{aid}={ACTION_NAMES.get(int(a), str(a))}({int(a)})"
            for aid, a in sorted(acts.items())
        )
        self.action_log_text.insert(self._tk.END, f"step {step_num}: {rendered}\n")

    def _redraw_agent_stats(self) -> None:
        state = self._active_state()
        t = self._theme
        yview = self.agent_stats_canvas.yview()
        self.agent_stats_canvas.delete("all")
        canvas_w = max(280, int(self.agent_stats_canvas.winfo_width() or 360))
        pad = 8
        content_w = canvas_w - (2 * pad)
        y = pad
        self.agent_stats_canvas.create_text(
            pad,
            y,
            anchor="nw",
            text="Agent Stats",
            fill=t["text"],
            font=("Courier", self._base_stats_font_size + 1, "bold"),
        )
        y += 24
        if state is None:
            self.agent_stats_canvas.create_text(
                pad,
                y,
                anchor="nw",
                text="No state loaded.",
                fill=t["text"],
                font=("Courier", self._base_stats_font_size),
            )
            self.agent_stats_canvas.configure(scrollregion=(0, 0, canvas_w, y + 32))
            self.agent_stats_canvas.yview_moveto(0.0)
            return

        alive_agents = sum(1 for a in state.agents.values() if a.alive)
        alive_monsters = sum(1 for m in state.monsters.values() if m.alive)
        self.agent_stats_canvas.create_text(
            pad,
            y,
            anchor="nw",
            text=(
                f"step={state.step_count}  "
                f"alive_agents={alive_agents}  "
                f"alive_monsters={alive_monsters}"
            ),
            fill=t["text_muted"],
            font=("Courier", self._base_stats_font_size),
        )
        y += 24

        for aid, agent in sorted(state.agents.items()):
            status = "alive" if agent.alive else "dead"
            card_h = 126
            self.agent_stats_canvas.create_rectangle(
                pad,
                y,
                pad + content_w,
                y + card_h,
                fill=t["bg"],
                outline=t["border"],
                width=1,
            )
            self.agent_stats_canvas.create_text(
                pad + 10,
                y + 8,
                anchor="nw",
                text=f"{aid} [{status}] {agent.profile_name}/{agent.race_name}/{agent.class_name}",
                fill=t["text"] if agent.alive else t["text_muted"],
                font=("Courier", self._base_stats_font_size, "bold"),
            )

            hp_ratio = 0.0
            if int(agent.max_hp) > 0:
                hp_ratio = max(0.0, min(1.0, float(agent.hp) / float(agent.max_hp)))
            bar_x0 = pad + 10
            bar_y0 = y + 30
            bar_w = max(80, content_w - 20)
            bar_h = 14
            self.agent_stats_canvas.create_rectangle(
                bar_x0,
                bar_y0,
                bar_x0 + bar_w,
                bar_y0 + bar_h,
                fill=t["danger"],
                outline=t["danger"],
            )
            self.agent_stats_canvas.create_rectangle(
                bar_x0,
                bar_y0,
                bar_x0 + int(bar_w * hp_ratio),
                bar_y0 + bar_h,
                fill=t["success"],
                outline=t["success"],
            )
            self.agent_stats_canvas.create_text(
                bar_x0 + 4,
                bar_y0 - 1,
                anchor="nw",
                text=f"HP {agent.hp}/{agent.max_hp}",
                fill=t["text"],
                font=("Courier", self._base_stats_font_size - 1, "bold"),
            )

            line1 = f"hunger={agent.hunger}/{agent.max_hunger}  pos={agent.position}  faction={int(agent.faction_id)}"
            line2 = (
                f"str={agent.strength} dex={agent.dexterity} int={agent.intellect}  "
                f"inv={len(agent.inventory)} eq={len(agent.equipped)}"
            )
            overall_level = int(
                sum(max(0, int(v)) for v in dict(agent.skills).values())
            )
            line2b = f"overall_level={overall_level}"
            top_skills = sorted(agent.skills.items(), key=lambda kv: (-int(kv[1]), kv[0]))[:4]
            line3 = "skills: " + ", ".join(f"{k}={int(v)}" for k, v in top_skills)
            self.agent_stats_canvas.create_text(
                bar_x0,
                y + 50,
                anchor="nw",
                text=line1,
                fill=t["text_muted"],
                font=("Courier", self._base_stats_font_size - 1),
            )
            self.agent_stats_canvas.create_text(
                bar_x0,
                y + 68,
                anchor="nw",
                text=line2,
                fill=t["text_muted"],
                font=("Courier", self._base_stats_font_size - 1),
            )
            self.agent_stats_canvas.create_text(
                bar_x0,
                y + 84,
                anchor="nw",
                text=line2b,
                fill=t["text_muted"],
                font=("Courier", self._base_stats_font_size - 1),
            )
            self.agent_stats_canvas.create_text(
                bar_x0,
                y + 100,
                anchor="nw",
                text=line3,
                fill=t["accent"],
                font=("Courier", self._base_stats_font_size - 1),
            )
            y += card_h + 8

        total_h = max(y + pad, int(self.agent_stats_canvas.winfo_height() or 1))
        self.agent_stats_canvas.configure(scrollregion=(0, 0, canvas_w, total_h))
        y0 = float(yview[0]) if yview else 0.0
        self.agent_stats_canvas.yview_moveto(max(0.0, min(1.0, y0)))

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
        action_log: Optional[List[Dict[str, object]]] = None,
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

    def restart(self) -> None:
        if not self._playback_states:
            return
        self._cursor = 0
        self.playback.pause()
        self._redraw()
        self.pump()

    def set_episode_navigation(
        self,
        on_prev_episode: Optional[Callable[[], None]] = None,
        on_next_episode: Optional[Callable[[], None]] = None,
    ) -> None:
        self._on_prev_episode = on_prev_episode
        self._on_next_episode = on_next_episode

    def prev_episode(self) -> None:
        if self._on_prev_episode is not None:
            self._on_prev_episode()

    def next_episode(self) -> None:
        if self._on_next_episode is not None:
            self._on_next_episode()

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
        if self._tooltip_window is not None:
            self._tooltip_window.destroy()
            self._tooltip_window = None
            self._tooltip_label = None
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
