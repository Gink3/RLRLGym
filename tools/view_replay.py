#!/usr/bin/env python3
"""PyQt6 replay viewer with smooth pan/zoom, selection, and playback controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlrlgym.content.tiles import load_tileset  # noqa: E402
from rlrlgym.world.replay_viewer_support import (  # noqa: E402
    build_activity_log_lines,
    collect_agent_ids,
    load_profile_map,
    resource_node_sprite_id,
    supported_animal_sprite_ids,
    supported_construction_sprite_ids,
    supported_resource_sprite_ids,
    visible_tiles_for_agent,
)


COLOR_MAP: Dict[str, str] = {
    "black": "#111111",
    "bright_black": "#4b4b4b",
    "red": "#d66b6b",
    "green": "#67b35c",
    "yellow": "#d6c15c",
    "blue": "#5c84d6",
    "magenta": "#b86bd6",
    "cyan": "#58bfbf",
    "white": "#e6e6e6",
    "bright_green": "#7ddf6a",
    "bright_red": "#f07b7b",
    "bright_yellow": "#f1d37a",
    "bright_white": "#f5f6fa",
}


class MapView(QGraphicsView):
    def __init__(self, scene: QGraphicsScene, parent: QWidget | None = None) -> None:
        super().__init__(scene, parent)
        self.tile_px = 24
        self._press_pos = None
        self._tile_click_cb = None
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.TextAntialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)

    def set_tile_click_callback(self, cb) -> None:
        self._tile_click_cb = cb

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.0015 ** delta
        self.scale(factor, factor)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_pos = event.position()
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._press_pos is not None
            and self._tile_click_cb is not None
        ):
            delta = event.position() - self._press_pos
            if abs(delta.x()) < 4 and abs(delta.y()) < 4:
                scene_pos = self.mapToScene(event.position().toPoint())
                col = int(scene_pos.x() // max(1, self.tile_px))
                row = int(scene_pos.y() // max(1, self.tile_px))
                self._tile_click_cb(row, col)
        self._press_pos = None
        super().mouseReleaseEvent(event)


class ReplayWindow(QMainWindow):
    def __init__(self, replay_path: Path, title: str) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1500, 940)

        self.replay_files = sorted(replay_path.parent.glob("*.replay.json"))
        if replay_path not in self.replay_files:
            self.replay_files.append(replay_path)
            self.replay_files.sort()
        self.replay_index = self.replay_files.index(replay_path)

        self.frames: List[dict] = []
        self.step_logs: List[dict] = []
        self.tile_defs = load_tileset("data/base/tiles.json")
        self.profile_defs = load_profile_map()
        self._animal_sprites = supported_animal_sprite_ids()
        self._resource_sprites = supported_resource_sprite_ids()
        self._construction_sprites = supported_construction_sprite_ids()
        self._sprite_cache: Dict[str, QPixmap] = {}
        self._current_frame_idx = 0
        self._selected_agent_id = ""
        self._visible_tiles: set[tuple[int, int]] = set()

        self.tile_px = 24
        self.playing = False

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        left_col = QVBoxLayout()
        layout.addLayout(left_col, stretch=5)

        self.scene = QGraphicsScene(self)
        self.view = MapView(self.scene, self)
        self.view.tile_px = self.tile_px
        self.view.set_tile_click_callback(self._on_map_tile_clicked)
        left_col.addWidget(self.view, stretch=1)

        controls = QHBoxLayout()
        self.prev_ep_btn = QPushButton("Prev Replay")
        self.next_ep_btn = QPushButton("Next Replay")
        self.prev_btn = QPushButton("Prev")
        self.play_btn = QPushButton("Play")
        self.next_btn = QPushButton("Next")
        self.frame_label = QLabel("frame 0/0")
        controls.addWidget(self.prev_ep_btn)
        controls.addWidget(self.next_ep_btn)
        controls.addWidget(self.prev_btn)
        controls.addWidget(self.play_btn)
        controls.addWidget(self.next_btn)
        controls.addWidget(self.frame_label)
        left_col.addLayout(controls)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)
        left_col.addWidget(self.slider)

        right_col = QVBoxLayout()
        layout.addLayout(right_col, stretch=2)

        right_col.addWidget(QLabel("Replay Files"))
        self.replay_list = QListWidget()
        right_col.addWidget(self.replay_list, stretch=1)

        right_col.addWidget(QLabel("Selected Agent / LOS"))
        self.agent_list = QListWidget()
        self.agent_list.setMaximumHeight(160)
        right_col.addWidget(self.agent_list)

        right_col.addWidget(QLabel("Frame Summary"))
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        right_col.addWidget(self.summary, stretch=1)

        right_col.addWidget(QLabel("Action / Events"))
        right_col.addWidget(QLabel("Log Agent Filter"))
        self.log_agent_list = QListWidget()
        self.log_agent_list.setMaximumHeight(140)
        right_col.addWidget(self.log_agent_list)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        right_col.addWidget(self.log_text, stretch=1)

        self.timer = QTimer(self)
        self.timer.setInterval(110)
        self.timer.timeout.connect(self._tick_playback)

        self.prev_btn.clicked.connect(lambda: self._set_frame(self.slider.value() - 1))
        self.next_btn.clicked.connect(lambda: self._set_frame(self.slider.value() + 1))
        self.play_btn.clicked.connect(self._toggle_play)
        self.slider.valueChanged.connect(self._set_frame)
        self.prev_ep_btn.clicked.connect(lambda: self._load_replay(self.replay_index - 1))
        self.next_ep_btn.clicked.connect(lambda: self._load_replay(self.replay_index + 1))
        self.replay_list.currentRowChanged.connect(self._load_replay)
        self.log_agent_list.itemChanged.connect(self._on_log_filter_changed)
        self.agent_list.currentRowChanged.connect(self._on_selected_agent_changed)

        self._reload_replay_list()
        self._load_replay(self.replay_index)

    def _reload_replay_list(self) -> None:
        self.replay_list.blockSignals(True)
        self.replay_list.clear()
        for p in self.replay_files:
            self.replay_list.addItem(QListWidgetItem(p.name))
        self.replay_list.setCurrentRow(self.replay_index)
        self.replay_list.blockSignals(False)

    def _load_replay(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.replay_files):
            return
        self.replay_index = idx
        path = self.replay_files[idx]
        payload = json.loads(path.read_text(encoding="utf-8"))
        frames_raw = payload.get("frames", [])
        if not isinstance(frames_raw, list) or not frames_raw:
            raise ValueError(f"No frames in replay: {path}")
        self.frames = [dict(row) for row in frames_raw if isinstance(row, dict)]
        step_logs = payload.get("step_logs", [])
        self.step_logs = [dict(row) for row in step_logs if isinstance(row, dict)] if isinstance(step_logs, list) else []
        self._rebuild_agent_lists()
        self.setWindowTitle(f"RLRLGym Replay Viewer - {path.name}")
        self.slider.blockSignals(True)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, len(self.frames) - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self._set_frame(0)
        self.replay_list.blockSignals(True)
        self.replay_list.setCurrentRow(idx)
        self.replay_list.blockSignals(False)

    def _rebuild_agent_lists(self) -> None:
        agent_ids = collect_agent_ids(self.frames[0] if self.frames else {}, self.step_logs)

        self.agent_list.blockSignals(True)
        self.agent_list.clear()
        self.agent_list.addItem(QListWidgetItem("(none)"))
        for aid in agent_ids:
            self.agent_list.addItem(QListWidgetItem(aid))
        if self._selected_agent_id and self._selected_agent_id in agent_ids:
            self.agent_list.setCurrentRow(agent_ids.index(self._selected_agent_id) + 1)
        else:
            self._selected_agent_id = ""
            self.agent_list.setCurrentRow(0)
        self.agent_list.blockSignals(False)

        self.log_agent_list.blockSignals(True)
        self.log_agent_list.clear()
        for aid in agent_ids:
            item = QListWidgetItem(aid)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.log_agent_list.addItem(item)
        self.log_agent_list.blockSignals(False)

    def _on_map_tile_clicked(self, row: int, col: int) -> None:
        if not self.frames:
            return
        agents = self.frames[self._current_frame_idx].get("agents", {})
        if not isinstance(agents, dict):
            return
        clicked_agent = ""
        for aid, data in sorted(agents.items()):
            if not isinstance(data, dict) or not bool(data.get("alive", True)):
                continue
            pos = data.get("position", [])
            if isinstance(pos, list) and len(pos) == 2 and int(pos[0]) == row and int(pos[1]) == col:
                clicked_agent = aid
                break
        if not clicked_agent:
            self.agent_list.setCurrentRow(0)
            return
        for idx in range(self.agent_list.count()):
            item = self.agent_list.item(idx)
            if item.text() == clicked_agent:
                self.agent_list.setCurrentRow(idx)
                return

    def _on_selected_agent_changed(self, row: int) -> None:
        item = self.agent_list.item(row) if row >= 0 else None
        text = item.text().strip() if item is not None else ""
        self._selected_agent_id = "" if text == "(none)" else text
        self._set_frame(self._current_frame_idx)

    def _on_log_filter_changed(self, _item: QListWidgetItem) -> None:
        self._refresh_log_text(self._current_frame_idx)

    def _selected_log_agents(self) -> set[str]:
        out: set[str] = set()
        for idx in range(self.log_agent_list.count()):
            item = self.log_agent_list.item(idx)
            if item.checkState() == Qt.CheckState.Checked:
                out.add(item.text().strip())
        return out

    def _refresh_log_text(self, idx: int) -> None:
        selected_agents = self._selected_log_agents()
        if idx > 0 and idx - 1 < len(self.step_logs):
            lines = build_activity_log_lines(self.step_logs[idx - 1], selected_agents)
            self.log_text.setPlainText(
                "\n".join(lines) if lines else "(no matching agent logs for current filter)"
            )
        else:
            self.log_text.setPlainText("(no step log for first frame)")

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        if self.playing:
            self.play_btn.setText("Pause")
            self.timer.start()
        else:
            self.play_btn.setText("Play")
            self.timer.stop()

    def _tick_playback(self) -> None:
        nxt = self.slider.value() + 1
        if nxt > self.slider.maximum():
            self.playing = False
            self.play_btn.setText("Play")
            self.timer.stop()
            return
        self.slider.setValue(nxt)

    def _tile_color(self, name: str) -> QColor:
        return QColor(COLOR_MAP.get(name, "#d0d0d0"))

    def _faction_bg_color(self, faction_id: int) -> QColor:
        if faction_id < 0:
            return QColor(90, 90, 90, 120)
        hue = (int(faction_id) * 67) % 360
        color = QColor.fromHsv(hue, 180, 220)
        color.setAlpha(120)
        return color

    def _entity_sprite(self, sprite_id: str) -> QPixmap:
        key = f"{sprite_id}:{self.tile_px}"
        if key in self._sprite_cache:
            return self._sprite_cache[key]
        pm = QPixmap(self.tile_px, self.tile_px)
        pm.fill(Qt.GlobalColor.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        center = self.tile_px // 2
        if sprite_id in self._animal_sprites:
            p.setBrush(QColor("#a7db8d"))
            p.setPen(QPen(QColor("#5f8c49"), 1))
            p.drawEllipse(5, 7, self.tile_px - 10, self.tile_px - 10)
            p.drawEllipse(4, 4, 5, 5)
            p.drawEllipse(self.tile_px - 9, 4, 5, 5)
            p.setBrush(QColor("#263238"))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawEllipse(center - 1, center + 1, 2, 2)
        elif sprite_id in self._resource_sprites:
            if sprite_id == "timber":
                p.setBrush(QColor("#8a6137"))
                p.setPen(QPen(QColor("#caa16f"), 2))
                p.drawRect(9, 11, 6, 10)
                p.setBrush(QColor("#3a8c47"))
                p.setPen(QPen(QColor("#2a6d38"), 1))
                p.drawEllipse(4, 4, 16, 10)
                p.drawEllipse(9, 2, 11, 11)
            elif sprite_id == "berries":
                p.setBrush(QColor("#3b8b44"))
                p.setPen(QPen(QColor("#2a6d38"), 1))
                p.drawEllipse(5, 7, 14, 11)
                p.setBrush(QColor("#c678dd"))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(8, 10, 4, 4)
                p.drawEllipse(13, 8, 4, 4)
                p.drawEllipse(14, 13, 4, 4)
            elif sprite_id == "grain":
                p.setPen(QPen(QColor("#d4b24a"), 2))
                p.drawLine(8, 20, 8, 8)
                p.drawLine(12, 20, 12, 6)
                p.drawLine(16, 20, 16, 9)
                p.drawLine(8, 10, 11, 8)
                p.drawLine(12, 8, 15, 6)
                p.drawLine(16, 11, 19, 9)
            elif sprite_id == "herb":
                p.setPen(QPen(QColor("#6dbd61"), 2))
                p.drawLine(center, 20, 8, 9)
                p.drawLine(center, 20, 16, 8)
                p.drawLine(center, 20, 12, 6)
            else:
                fill = {
                    "stone": "#9ba7b4",
                    "clay": "#b88462",
                    "flint": "#808998",
                    "ore": "#cda14a",
                    "generic": "#84b3d9",
                }.get(sprite_id, "#84b3d9")
                p.setBrush(QColor(fill))
                p.setPen(QPen(QColor("#4e5965"), 1))
                p.drawEllipse(5, 10, 8, 8)
                p.drawEllipse(11, 7, 8, 9)
                p.drawEllipse(14, 12, 5, 5)
        else:
            if sprite_id in {"wood_wall", "rock_wall"}:
                p.setBrush(QColor("#6c5137" if sprite_id == "wood_wall" else "#57616d"))
                p.setPen(QPen(QColor("#2d333b"), 1))
                p.drawRect(4, 4, self.tile_px - 8, self.tile_px - 8)
                if sprite_id == "wood_wall":
                    p.setPen(QPen(QColor("#8d6846"), 2))
                    for xx in (8, 12, 16):
                        p.drawLine(xx, 5, xx, self.tile_px - 5)
                else:
                    p.setPen(QPen(QColor("#8691a0"), 1))
                    p.drawLine(6, 10, self.tile_px - 6, 10)
                    p.drawLine(10, 6, 10, self.tile_px - 6)
            elif sprite_id == "wood_door":
                p.setBrush(QColor("#7c5937"))
                p.setPen(QPen(QColor("#4d331c"), 1))
                p.drawRoundedRect(6, 3, self.tile_px - 12, self.tile_px - 6, 2, 2)
                p.setBrush(QColor("#d7b25c"))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(self.tile_px - 10, center, 3, 3)
            elif sprite_id == "glass_window":
                p.setBrush(QColor(129, 205, 230, 170))
                p.setPen(QPen(QColor("#d7efe9"), 1))
                p.drawRect(5, 5, self.tile_px - 10, self.tile_px - 10)
                p.drawLine(center, 6, center, self.tile_px - 6)
                p.drawLine(6, center, self.tile_px - 6, center)
            elif sprite_id == "spike_trap":
                p.setPen(QPen(QColor("#d89e71"), 2))
                for xx in (6, 10, 14, 18):
                    p.drawLine(xx, 20, xx + 2, 10)
            elif sprite_id in {"campfire", "firepit", "fireplace"}:
                p.setPen(QPen(QColor("#7a5636"), 2))
                p.drawLine(7, 18, 12, 13)
                p.drawLine(17, 18, 12, 13)
                p.drawLine(9, 19, 15, 11)
                p.setBrush(QColor("#ffb347" if sprite_id != "fireplace" else "#fff1b0"))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(9, 7, 7, 10)
                if sprite_id == "firepit":
                    p.setPen(QPen(QColor("#9ba7b4"), 2))
                    p.drawEllipse(6, 12, 12, 8)
            elif sprite_id == "workbench":
                p.setBrush(QColor("#8a6137"))
                p.setPen(QPen(QColor("#5f4324"), 1))
                p.drawRect(5, 8, self.tile_px - 10, 5)
                p.drawRect(7, 13, 3, 8)
                p.drawRect(self.tile_px - 10, 13, 3, 8)
            elif sprite_id in {"clay_forge", "clay_furnace", "clay_smelter"}:
                base = {"clay_forge": "#9b7453", "clay_furnace": "#8b6a4e", "clay_smelter": "#7f6148"}[sprite_id]
                p.setBrush(QColor(base))
                p.setPen(QPen(QColor("#4f3b2d"), 1))
                p.drawRoundedRect(4, 5, self.tile_px - 8, self.tile_px - 8, 3, 3)
                p.setBrush(QColor("#f4c06a"))
                p.setPen(Qt.PenStyle.NoPen)
                p.drawEllipse(9, 11, 6, 6)
            else:
                p.setBrush(QColor("#88a2b8"))
                p.setPen(QPen(QColor("#4b5d6e"), 1))
                p.drawRoundedRect(5, 5, self.tile_px - 10, self.tile_px - 10, 2, 2)
        p.end()
        self._sprite_cache[key] = pm
        return pm

    def _add_sprite(self, sprite_id: str, row: int, col: int, z: float) -> None:
        item = QGraphicsPixmapItem(self._entity_sprite(sprite_id))
        item.setPos(col * self.tile_px, row * self.tile_px)
        item.setZValue(z)
        self.scene.addItem(item)

    def _selected_agent_visible_tiles(self, frame: dict) -> set[tuple[int, int]]:
        if not self._selected_agent_id:
            return set()
        return visible_tiles_for_agent(
            frame,
            self._selected_agent_id,
            self.tile_defs,
            self.profile_defs,
        )

    def _set_frame(self, idx: int) -> None:
        if not self.frames:
            return
        idx = max(0, min(idx, len(self.frames) - 1))
        frame = self.frames[idx]
        grid = frame.get("grid", [])
        if not isinstance(grid, list):
            return

        self.scene.clear()
        h = len(grid)
        w = len(grid[0]) if h > 0 and isinstance(grid[0], list) else 0
        self._visible_tiles = self._selected_agent_visible_tiles(frame)

        for r, row in enumerate(grid):
            if not isinstance(row, list):
                continue
            for c, tile_id_any in enumerate(row):
                tile_id = str(tile_id_any)
                td = self.tile_defs.get(tile_id)
                glyph = td.glyph if td is not None else "?"
                color_name = td.color if td is not None else "white"
                x = c * self.tile_px
                y = r * self.tile_px
                bg = QColor("#1f232a") if (r + c) % 2 == 0 else QColor("#242a33")
                self.scene.addRect(x, y, self.tile_px, self.tile_px, QPen(Qt.PenStyle.NoPen), bg)
                if self._selected_agent_id:
                    if (r, c) in self._visible_tiles:
                        self.scene.addRect(
                            x + 1,
                            y + 1,
                            self.tile_px - 2,
                            self.tile_px - 2,
                            QPen(Qt.PenStyle.NoPen),
                            QColor(73, 155, 114, 38),
                        )
                    else:
                        self.scene.addRect(
                            x,
                            y,
                            self.tile_px,
                            self.tile_px,
                            QPen(Qt.PenStyle.NoPen),
                            QColor(8, 10, 12, 110),
                        )
                if tile_id in self._construction_sprites:
                    self._add_sprite(tile_id, r, c, 1.0)
                else:
                    text = QGraphicsSimpleTextItem(glyph)
                    text.setBrush(self._tile_color(color_name))
                    text.setFont(QFont("DejaVu Sans Mono", 10))
                    text.setPos(x + 5, y + 2)
                    text.setZValue(1.0)
                    self.scene.addItem(text)

        for row in frame.get("plant_plots", []):
            if not isinstance(row, dict):
                continue
            pos = row.get("position", [])
            if not isinstance(pos, list) or len(pos) != 2:
                continue
            r = int(pos[0])
            c = int(pos[1])
            x = c * self.tile_px
            y = r * self.tile_px
            outline = QPen(QColor("#44d1d1"))
            outline.setWidth(2)
            rect = QGraphicsRectItem(x + 1, y + 1, self.tile_px - 2, self.tile_px - 2)
            rect.setPen(outline)
            rect.setZValue(1.8)
            self.scene.addItem(rect)

        for row in frame.get("resource_nodes", []):
            if not isinstance(row, dict):
                continue
            pos = row.get("position", [])
            if not isinstance(pos, list) or len(pos) != 2:
                continue
            self._add_sprite(
                resource_node_sprite_id(
                    node_id=str(row.get("node_id", "")),
                    drop_item=str(row.get("drop_item", "")),
                ),
                int(pos[0]),
                int(pos[1]),
                2.0,
            )

        for row in frame.get("chests", []):
            if not isinstance(row, dict):
                continue
            pos = row.get("position", [])
            if not isinstance(pos, list) or len(pos) != 2:
                continue
            self._add_sprite("chest", int(pos[0]), int(pos[1]), 2.1)
            if bool(row.get("opened", False)):
                lid = QGraphicsSimpleTextItem("open")
                lid.setBrush(QColor("#f0d060"))
                lid.setFont(QFont("DejaVu Sans Mono", 6))
                lid.setPos(int(pos[1]) * self.tile_px + 2, int(pos[0]) * self.tile_px + 15)
                lid.setZValue(2.2)
                self.scene.addItem(lid)

        for row in frame.get("stations", []):
            if not isinstance(row, dict):
                continue
            pos = row.get("position", [])
            if not isinstance(pos, list) or len(pos) != 2:
                continue
            self._add_sprite(str(row.get("station_id", "workbench")), int(pos[0]), int(pos[1]), 2.3)

        for row in frame.get("monsters", []):
            if not isinstance(row, dict) or not bool(row.get("alive", True)):
                continue
            pos = row.get("position", [])
            if not isinstance(pos, list) or len(pos) != 2:
                continue
            r = int(pos[0])
            c = int(pos[1])
            m = QGraphicsSimpleTextItem(str(row.get("symbol", "M"))[:1] or "M")
            m.setBrush(QColor("#d66b6b"))
            m.setFont(QFont("DejaVu Sans Mono", 11, QFont.Weight.Bold))
            m.setPos(c * self.tile_px + 5, r * self.tile_px + 2)
            m.setZValue(3.0)
            self.scene.addItem(m)

        for row in frame.get("animals", []):
            if not isinstance(row, dict) or not bool(row.get("alive", True)):
                continue
            pos = row.get("position", [])
            if not isinstance(pos, list) or len(pos) != 2:
                continue
            self._add_sprite(str(row.get("animal_id", "")) or "rabbit", int(pos[0]), int(pos[1]), 3.1)

        agents = frame.get("agents", {})
        selected_pos = None
        if isinstance(agents, dict):
            for aid, row in sorted(agents.items()):
                if not isinstance(row, dict) or not bool(row.get("alive", True)):
                    continue
                pos = row.get("position", [])
                if not isinstance(pos, list) or len(pos) != 2:
                    continue
                r = int(pos[0])
                c = int(pos[1])
                x = c * self.tile_px
                y = r * self.tile_px
                faction_id = int(row.get("faction_id", -1))
                self.scene.addRect(x, y, self.tile_px, self.tile_px, QPen(Qt.PenStyle.NoPen), self._faction_bg_color(faction_id))
                label = aid.split("_")[-1]
                t = QGraphicsSimpleTextItem(label)
                t.setBrush(QColor("#f3f7ff"))
                t.setFont(QFont("DejaVu Sans Mono", 11, QFont.Weight.Bold))
                t.setPos(x + 4, y + 2)
                t.setZValue(4.0)
                self.scene.addItem(t)
                if aid == self._selected_agent_id:
                    selected_pos = (r, c)
                    outline = QPen(QColor("#ffd166"))
                    outline.setWidth(2)
                    self.scene.addRect(x + 1, y + 1, self.tile_px - 2, self.tile_px - 2, outline)
        if selected_pos is not None:
            cx = selected_pos[1] * self.tile_px + self.tile_px / 2.0
            cy = selected_pos[0] * self.tile_px + self.tile_px / 2.0
            for rr, cc in sorted(self._visible_tiles):
                tx = cc * self.tile_px + self.tile_px / 2.0
                ty = rr * self.tile_px + self.tile_px / 2.0
                if abs(tx - cx) + abs(ty - cy) <= self.tile_px:
                    continue
                sight_pen = QPen(QColor(255, 209, 102, 48))
                sight_pen.setWidth(1)
                self.scene.addLine(cx, cy, tx, ty, sight_pen)

        self.scene.setSceneRect(0, 0, max(1, w * self.tile_px), max(1, h * self.tile_px))
        self.frame_label.setText(f"frame {idx + 1}/{len(self.frames)}")
        self._current_frame_idx = idx

        agents_count = len(frame.get("agents", {})) if isinstance(frame.get("agents", {}), dict) else 0
        monsters_count = len([x for x in frame.get("monsters", []) if isinstance(x, dict) and bool(x.get("alive", True))])
        animals_count = len([x for x in frame.get("animals", []) if isinstance(x, dict) and bool(x.get("alive", True))])
        plants_count = len(frame.get("plant_plots", [])) if isinstance(frame.get("plant_plots", []), list) else 0
        resource_nodes_count = len(frame.get("resource_nodes", [])) if isinstance(frame.get("resource_nodes", []), list) else 0
        stations_count = len(frame.get("stations", [])) if isinstance(frame.get("stations", []), list) else 0
        summary_lines = [
            f"size: {w}x{h}",
            f"agents: {agents_count}",
            f"monsters(alive): {monsters_count}",
            f"animals(alive): {animals_count}",
            f"resource_nodes: {resource_nodes_count}",
            f"plant_plots: {plants_count}",
            f"stations: {stations_count}",
            f"selected_agent: {self._selected_agent_id or '(none)'}",
            f"selected_agent_visible_tiles: {len(self._visible_tiles)}",
        ]
        self.summary.setPlainText("\n".join(summary_lines))

        self._refresh_log_text(idx)


def main() -> None:
    parser = argparse.ArgumentParser(description="View a saved episode replay (PyQt6)")
    parser.add_argument("replay_path", type=str, help="Path to *.replay.json file")
    parser.add_argument("--title", type=str, default="RLRLGym Replay Viewer")
    args = parser.parse_args()

    replay_path = Path(args.replay_path).resolve()
    if not replay_path.exists():
        raise FileNotFoundError(replay_path)

    app = QApplication(sys.argv)
    win = ReplayWindow(replay_path=replay_path, title=args.title)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
