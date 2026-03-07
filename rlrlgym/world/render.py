"""PyQt6 window-first renderer and playback controls."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

try:  # pragma: no cover - import availability depends on host environment
    from PyQt6.QtCore import QTimer, Qt
    from PyQt6.QtGui import QAction, QColor, QFont, QPainter, QPen
    from PyQt6.QtWidgets import (
        QApplication,
        QComboBox,
        QGraphicsRectItem,
        QGraphicsScene,
        QGraphicsSimpleTextItem,
        QGraphicsView,
        QHBoxLayout,
        QLabel,
        QMainWindow,
        QMenu,
        QPushButton,
        QSlider,
        QTextEdit,
        QToolButton,
        QVBoxLayout,
        QWidget,
    )
    _PYQT6_AVAILABLE = True
    _PYQT6_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover
    QApplication = None  # type: ignore[assignment]
    QComboBox = object  # type: ignore[assignment]
    QGraphicsRectItem = object  # type: ignore[assignment]
    QGraphicsScene = object  # type: ignore[assignment]
    QGraphicsSimpleTextItem = object  # type: ignore[assignment]
    QGraphicsView = object  # type: ignore[assignment]
    QHBoxLayout = object  # type: ignore[assignment]
    QLabel = object  # type: ignore[assignment]
    QMainWindow = object  # type: ignore[assignment]
    QMenu = object  # type: ignore[assignment]
    QPushButton = object  # type: ignore[assignment]
    QSlider = object  # type: ignore[assignment]
    QTextEdit = object  # type: ignore[assignment]
    QToolButton = object  # type: ignore[assignment]
    QVBoxLayout = object  # type: ignore[assignment]
    QWidget = object  # type: ignore[assignment]
    QTimer = object  # type: ignore[assignment]
    Qt = object  # type: ignore[assignment]
    QAction = object  # type: ignore[assignment]
    QColor = object  # type: ignore[assignment]
    QFont = object  # type: ignore[assignment]
    QPainter = object  # type: ignore[assignment]
    QPen = object  # type: ignore[assignment]
    _PYQT6_AVAILABLE = False
    _PYQT6_IMPORT_ERROR = exc

from ..systems.models import EnvState, TileDef
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
    "reward_explorer_policy_v1": ("E", "#3498db"),
    "reward_brawler_policy_v1": ("B", "#2ecc71"),
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
    ) -> List[List[Tuple[str, str, bool, bool]]]:
        min_r, max_r, min_c, max_c = self.view_bounds(
            state=state, focus_agent=focus_agent, zoom=zoom
        )

        agent_positions = {a.position: aid for aid, a in state.agents.items() if a.alive}
        chest_positions = {pos for pos, chest in state.chests.items() if not chest.opened}
        station_positions = {pos: station for pos, station in state.stations.items()}
        animal_positions = {animal.position: animal for animal in state.animals.values() if animal.alive}
        resource_positions = {
            pos: node
            for pos, node in state.resource_nodes.items()
            if int(node.remaining) > 0
        }
        monster_positions = {
            monster.position: monster
            for monster in state.monsters.values()
            if monster.alive
        }

        cells: List[List[Tuple[str, str, bool, bool]]] = []
        for r in range(min_r, max_r + 1):
            row_cells: List[Tuple[str, str, bool, bool]] = []
            for c in range(min_c, max_c + 1):
                pos = (r, c)
                if pos in agent_positions:
                    agent_id = agent_positions[pos]
                    agent = state.agents[agent_id]
                    symbol, color = PROFILE_AGENT_STYLE.get(
                        agent.profile_name, DEFAULT_AGENT_STYLE
                    )
                    row_cells.append((symbol, color, True, False))
                    continue
                if pos in monster_positions:
                    monster = monster_positions[pos]
                    row_cells.append((monster.symbol, TK_COLORS.get(monster.color, "#ff7675"), False, True))
                    continue
                if pos in animal_positions:
                    animal = animal_positions[pos]
                    row_cells.append((animal.symbol, TK_COLORS.get(animal.color, "#f6e58d"), False, False))
                    continue
                if pos in station_positions:
                    row_cells.append(("T", TK_COLORS["cyan"], False, False))
                    continue
                if pos in resource_positions:
                    node = resource_positions[pos]
                    glyph = "R"
                    color = TK_COLORS["green"]
                    if str(node.skill) == "mining":
                        glyph = "M"
                        color = TK_COLORS["yellow"]
                    elif str(node.skill) == "woodcutting":
                        glyph = "W"
                        color = TK_COLORS["green"]
                    row_cells.append((glyph, color, False, False))
                    continue
                if pos in chest_positions:
                    row_cells.append(("C", TK_COLORS["yellow"], False, False))
                    continue
                tile = self.tiles[state.grid[r][c]]
                row_cells.append((tile.glyph, TK_COLORS.get(tile.color, TK_COLORS["white"]), False, False))
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


class _VarCompat:
    def __init__(self, value: object, on_change: Callable[[], None] | None = None) -> None:
        self._value = value
        self._on_change = on_change

    def get(self) -> object:
        return self._value

    def set(self, value: object) -> None:
        self._value = value
        if self._on_change is not None:
            self._on_change()


class _RootProxy:
    def __init__(self, win: QMainWindow) -> None:
        self._win = win

    def title(self, text: str) -> None:
        self._win.setWindowTitle(str(text))


class _MapView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.TextAntialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.0015 ** delta
        self.scale(factor, factor)


class RenderWindow:
    """PyQt6 viewer for rendering states in a dedicated window."""

    def __init__(self, tiles: Dict[str, TileDef], title: str = "RLRLGym Viewer") -> None:
        if not _PYQT6_AVAILABLE:
            raise RuntimeError(
                "PyQt6 is required for window rendering"
            ) from _PYQT6_IMPORT_ERROR
        self._app = QApplication.instance()
        self._owns_app = False
        if self._app is None:
            self._app = QApplication(sys.argv)
            self._owns_app = True

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
        self._on_prev_episode: Optional[Callable[[], None]] = None
        self._on_next_episode: Optional[Callable[[], None]] = None

        self._window = QMainWindow()
        self._window.setWindowTitle(title)
        self._window.resize(1400, 900)
        self.root = _RootProxy(self._window)

        self.focus_var = _VarCompat("all", self._on_external_focus_change)
        self.zoom_var = _VarCompat(0, self._on_external_zoom_change)

        root = QWidget(self._window)
        self._window.setCentralWidget(root)
        layout = QVBoxLayout(root)

        controls = QHBoxLayout()
        layout.addLayout(controls)

        self.play_pause_btn = QPushButton("Play")
        self.restart_btn = QPushButton("Restart")
        self.step_btn = QPushButton("Step")
        self.prev_ep_btn = QPushButton("Prev Ep")
        self.next_ep_btn = QPushButton("Next Ep")
        self.speed_1x_btn = QPushButton("1x")
        self.speed_2x_btn = QPushButton("2x")
        self.speed_5x_btn = QPushButton("5x")

        controls.addWidget(self.play_pause_btn)
        controls.addWidget(self.restart_btn)
        controls.addWidget(self.step_btn)
        controls.addWidget(self.prev_ep_btn)
        controls.addWidget(self.next_ep_btn)
        controls.addWidget(self.speed_1x_btn)
        controls.addWidget(self.speed_2x_btn)
        controls.addWidget(self.speed_5x_btn)

        controls.addWidget(QLabel("Focus"))
        self.focus_combo = QComboBox()
        self.focus_combo.addItems(self._focus_choices)
        controls.addWidget(self.focus_combo)

        controls.addWidget(QLabel("Zoom"))
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setMinimum(0)
        self.zoom_slider.setMaximum(10)
        self.zoom_slider.setValue(0)
        self.zoom_slider.setMaximumWidth(180)
        controls.addWidget(self.zoom_slider)

        controls.addStretch(1)

        self.settings_button = QToolButton()
        self.settings_button.setText("Settings")
        self.settings_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.settings_menu = QMenu(self._window)
        self.theme_menu = self.settings_menu.addMenu("Theme")
        self._theme_actions: Dict[str, QAction] = {}
        for name in list_theme_names():
            action = QAction(theme_label(name), self._window)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, n=name: self._on_theme_change(n, checked))
            self.theme_menu.addAction(action)
            self._theme_actions[name] = action
        self.settings_button.setMenu(self.settings_menu)
        controls.addWidget(self.settings_button)

        body = QHBoxLayout()
        layout.addLayout(body, stretch=1)

        left = QVBoxLayout()
        body.addLayout(left, stretch=4)

        self.scene = QGraphicsScene(self._window)
        self.view = _MapView(self.scene, self._window)
        left.addWidget(self.view, stretch=1)

        self.frame_label = QLabel("frame: live")
        left.addWidget(self.frame_label)

        right = QVBoxLayout()
        body.addLayout(right, stretch=2)

        right.addWidget(QLabel("Action Log"))
        self.action_log = QTextEdit(self._window)
        self.action_log.setReadOnly(True)
        self.action_log.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        right.addWidget(self.action_log, stretch=1)

        right.addWidget(QLabel("Agent Stats"))
        self.agent_stats = QTextEdit(self._window)
        self.agent_stats.setReadOnly(True)
        self.agent_stats.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        right.addWidget(self.agent_stats, stretch=1)

        self.timer = QTimer(self._window)
        self.timer.timeout.connect(self._tick)

        self.play_pause_btn.clicked.connect(self.toggle_play_pause)
        self.restart_btn.clicked.connect(self.restart)
        self.step_btn.clicked.connect(self.step)
        self.prev_ep_btn.clicked.connect(self.prev_episode)
        self.next_ep_btn.clicked.connect(self.next_episode)
        self.speed_1x_btn.clicked.connect(self.playback.set_speed_1x)
        self.speed_2x_btn.clicked.connect(self.playback.set_speed_2x)
        self.speed_5x_btn.clicked.connect(self.playback.set_speed_5x)
        self.focus_combo.currentTextChanged.connect(self._on_focus_change)
        self.zoom_slider.valueChanged.connect(self._on_zoom_change)

        current = self._theme_name if self._theme_name in self._theme_actions else ""
        if current:
            self._theme_actions[current].setChecked(True)
        self._apply_theme()
        self._window.show()
        self.pump()

    def _on_theme_change(self, name: str, checked: bool) -> None:
        if not checked:
            return
        saved = save_selected_theme(name)
        self._theme_name = saved
        self._theme = get_theme(saved)
        for key, action in self._theme_actions.items():
            action.setChecked(key == saved)
        self._apply_theme()
        self._redraw()

    def _apply_theme(self) -> None:
        t = self._theme
        self._window.setStyleSheet(
            f"""
            QMainWindow {{ background: {t['bg']}; color: {t['text']}; }}
            QWidget {{ background: {t['panel']}; color: {t['text']}; }}
            QComboBox, QTextEdit {{
                background: {t['panel_alt']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 4px;
                padding: 4px;
            }}
            QPushButton, QToolButton {{
                background: {t['panel_alt']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 4px;
                padding: 5px 8px;
            }}
            QPushButton:hover, QToolButton:hover {{
                background: {t['primary_hover']};
            }}
            QMenu {{
                background: {t['panel']};
                color: {t['text']};
                border: 1px solid {t['border']};
            }}
            QMenu::item:selected {{
                background: {t['primary']};
            }}
            """
        )

    def _current_focus(self) -> Optional[str]:
        focus = str(self.focus_var.get()).strip()
        return None if not focus or focus == "all" else focus

    def _on_external_focus_change(self) -> None:
        focus = str(self.focus_var.get())
        idx = self.focus_combo.findText(focus)
        if idx >= 0:
            self.focus_combo.blockSignals(True)
            self.focus_combo.setCurrentIndex(idx)
            self.focus_combo.blockSignals(False)
        self._redraw()

    def _on_external_zoom_change(self) -> None:
        try:
            z = int(self.zoom_var.get())
        except Exception:
            z = 0
        z = max(0, min(10, z))
        self.zoom_var._value = z
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(z)
        self.zoom_slider.blockSignals(False)
        self._redraw()

    def _on_focus_change(self, _value: str) -> None:
        self.focus_var._value = self.focus_combo.currentText()
        self._redraw()

    def _on_zoom_change(self, value: int) -> None:
        self.zoom_var._value = int(value)
        self._redraw()

    def _set_focus_choices(self, focus_choices: Optional[List[str]]) -> None:
        values = ["all"]
        if focus_choices:
            values += [str(a) for a in focus_choices]
        current = str(self.focus_var.get())
        self._focus_choices = values
        self.focus_combo.blockSignals(True)
        self.focus_combo.clear()
        self.focus_combo.addItems(self._focus_choices)
        idx = self.focus_combo.findText(current)
        if idx < 0:
            idx = 0
            self.focus_var._value = "all"
        self.focus_combo.setCurrentIndex(idx)
        self.focus_combo.blockSignals(False)

    def _active_state(self) -> Optional[EnvState]:
        if self._playback_states:
            if self._cursor < 0:
                self._cursor = 0
            if self._cursor >= len(self._playback_states):
                self._cursor = len(self._playback_states) - 1
            return self._playback_states[self._cursor]
        return self._live_state

    def _tile_color(self, name: str) -> QColor:
        return QColor(TK_COLORS.get(name, TK_COLORS["white"]))

    def _redraw(self) -> None:
        state = self._active_state()
        if state is None:
            return

        self.scene.clear()
        focus = self._current_focus()
        zoom = int(self.zoom_var.get())
        cells = self.renderer.render_cells(state, focus_agent=focus, zoom=zoom)

        tile_px = 24
        for r, row in enumerate(cells):
            for c, (glyph, color, is_agent, is_monster) in enumerate(row):
                x = c * tile_px
                y = r * tile_px
                bg = QColor("#1e232b") if (r + c) % 2 == 0 else QColor("#242b35")
                rect = QGraphicsRectItem(x, y, tile_px, tile_px)
                rect.setPen(QPen(Qt.PenStyle.NoPen))
                rect.setBrush(bg)
                self.scene.addItem(rect)

                text = QGraphicsSimpleTextItem(glyph)
                text.setBrush(self._tile_color(color))
                weight = QFont.Weight.Bold if (is_agent or is_monster) else QFont.Weight.Normal
                text.setFont(QFont("DejaVu Sans Mono", 10, weight))
                text.setPos(x + 5, y + 2)
                self.scene.addItem(text)

        self.scene.setSceneRect(0, 0, max(1, len(cells[0]) * tile_px), max(1, len(cells) * tile_px))
        self._redraw_action_log()
        self._redraw_agent_stats(state)

    def _redraw_action_log(self) -> None:
        if not self._playback_states:
            self.frame_label.setText("frame: live")
            self.action_log.setPlainText("live mode")
            return

        total = len(self._playback_states)
        self.frame_label.setText(f"frame {self._cursor + 1}/{total}")
        idx = min(self._cursor, len(self._playback_actions) - 1)
        if idx >= 0 and idx < len(self._playback_actions):
            payload = self._playback_actions[idx]
            self.action_log.setPlainText(json.dumps(payload, indent=2, sort_keys=True))
        else:
            self.action_log.setPlainText("(no action log for this frame)")

    def _redraw_agent_stats(self, state: EnvState) -> None:
        lines: List[str] = []
        for aid in sorted(state.agents.keys()):
            a = state.agents[aid]
            lines.append(
                f"{aid} hp={a.hp}/{a.max_hp} hunger={a.hunger:.2f} pos={a.position} "
                f"alive={a.alive} race={a.race_name} class={a.class_name} faction={a.faction_id}"
            )
        alive_monsters = sum(1 for m in state.monsters.values() if m.alive)
        alive_animals = sum(1 for a in state.animals.values() if a.alive)
        lines.append("")
        lines.append(f"step={state.step_count} monsters_alive={alive_monsters} animals_alive={alive_animals}")
        self.agent_stats.setPlainText("\n".join(lines))

    def update_state(self, state: EnvState, focus_choices: Optional[List[str]] = None) -> None:
        self._live_state = state
        if focus_choices is not None:
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
        self._playback_states = list(states)
        self._playback_actions = list(action_log or [])
        self._cursor = 0
        self._playing = False
        self.playback.pause()
        self.play_pause_btn.setText("Play")
        self.timer.stop()
        if focus_choices is not None:
            self._set_focus_choices(focus_choices)
        self._redraw()
        self.pump()

    def clear_playback(self) -> None:
        self._playback_states = []
        self._playback_actions = []
        self._cursor = 0
        self._playing = False
        self.playback.pause()
        self.play_pause_btn.setText("Play")
        self.timer.stop()
        self._redraw()

    def toggle_play_pause(self) -> None:
        if self._playing:
            self.pause()
        else:
            self.play()

    def play(self) -> None:
        self._playing = True
        self.playback.play()
        self.play_pause_btn.setText("Pause")
        interval = int(max(20, 200 / max(0.25, self.playback.speed)))
        self.timer.start(interval)

    def pause(self) -> None:
        self._playing = False
        self.playback.pause()
        self.play_pause_btn.setText("Play")
        self.timer.stop()

    def step(self) -> None:
        if not self._playback_states:
            return
        self.pause()
        if self._cursor < len(self._playback_states) - 1:
            self._cursor += 1
            self._redraw()
        self.pump()

    def restart(self) -> None:
        if not self._playback_states:
            return
        self.pause()
        self._cursor = 0
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
        if not self._playing or not self._playback_states:
            return
        if self._cursor >= len(self._playback_states) - 1:
            self.pause()
            return
        self._cursor += 1
        self._redraw()
        self.pump()

    def pump(self) -> None:
        assert self._app is not None
        self._app.processEvents()

    def close(self) -> None:
        self.pause()
        self._window.close()

    def run(self) -> None:
        self._window.show()
        if self._owns_app:
            assert self._app is not None
            _ = self._app.exec()
        else:
            self.pump()
