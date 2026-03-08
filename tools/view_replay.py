#!/usr/bin/env python3
"""PyQt6 replay viewer with smooth pan/zoom and playback controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
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

from rlrlgym.world.env import PLANT_TYPES  # noqa: E402
from rlrlgym.content.tiles import load_tileset  # noqa: E402


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
}


class MapView(QGraphicsView):
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
        # Continuous exponential scaling gives smoother zoom than fixed jumps.
        factor = 1.0015 ** delta
        self.scale(factor, factor)


class ReplayWindow(QMainWindow):
    def __init__(self, replay_path: Path, title: str) -> None:
        super().__init__()
        self.setWindowTitle(title)
        self.resize(1400, 900)

        self.replay_files = sorted(replay_path.parent.glob("*.replay.json"))
        if replay_path not in self.replay_files:
            self.replay_files.append(replay_path)
            self.replay_files.sort()
        self.replay_index = self.replay_files.index(replay_path)

        self.frames: List[dict] = []
        self.step_logs: List[dict] = []
        self.tile_defs = load_tileset("data/base/tiles.json")

        self.tile_px = 24
        self.playing = False

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        left_col = QVBoxLayout()
        layout.addLayout(left_col, stretch=4)

        self.scene = QGraphicsScene(self)
        self.view = MapView(self.scene, self)
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

        right_col.addWidget(QLabel("Frame Summary"))
        self.summary = QTextEdit()
        self.summary.setReadOnly(True)
        right_col.addWidget(self.summary, stretch=1)

        right_col.addWidget(QLabel("Action / Events"))
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

    def _resource_node_style(self, node_id: str, drop_item: str) -> tuple[str, QColor]:
        node = str(node_id).strip().lower()
        drop = str(drop_item).strip().lower()
        if "timber" in node:
            return "T", QColor("#c5904a")
        if "berry" in node:
            return "B", QColor("#d28be0")
        if "grain" in node:
            return "G", QColor("#e3c66d")
        if "herb" in node:
            return "H", QColor("#79c66a")
        if drop in {"stone", "clay", "flint"}:
            return "R", QColor("#aeb9c7")
        if drop.endswith("_ore"):
            return "O", QColor("#d9b062")
        return "N", QColor("#9bc4df")

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

        # Base tiles
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
                text = QGraphicsSimpleTextItem(glyph)
                text.setBrush(self._tile_color(color_name))
                text.setFont(QFont("DejaVu Sans Mono", 10))
                text.setPos(x + 5, y + 2)
                self.scene.addItem(text)

        # Plant plots highlight
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
            self.scene.addRect(x + 1, y + 1, self.tile_px - 2, self.tile_px - 2, outline)

        # Resource nodes
        for row in frame.get("resource_nodes", []):
            if not isinstance(row, dict):
                continue
            pos = row.get("position", [])
            if not isinstance(pos, list) or len(pos) != 2:
                continue
            r = int(pos[0])
            c = int(pos[1])
            x = c * self.tile_px
            y = r * self.tile_px
            glyph, color = self._resource_node_style(
                node_id=str(row.get("node_id", "")),
                drop_item=str(row.get("drop_item", "")),
            )
            t = QGraphicsSimpleTextItem(glyph)
            t.setBrush(color)
            t.setFont(QFont("DejaVu Sans Mono", 11, QFont.Weight.Bold))
            t.setPos(x + 4, y + 2)
            self.scene.addItem(t)

        # Chests
        for row in frame.get("chests", []):
            if not isinstance(row, dict):
                continue
            pos = row.get("position", [])
            if not isinstance(pos, list) or len(pos) != 2:
                continue
            r = int(pos[0])
            c = int(pos[1])
            x = c * self.tile_px
            y = r * self.tile_px
            ch = QGraphicsSimpleTextItem("C" if not bool(row.get("opened", False)) else "c")
            ch.setBrush(QColor("#f0d060"))
            ch.setFont(QFont("DejaVu Sans Mono", 11, QFont.Weight.Bold))
            ch.setPos(x + 5, y + 2)
            self.scene.addItem(ch)

        # Monsters
        for row in frame.get("monsters", []):
            if not isinstance(row, dict) or not bool(row.get("alive", True)):
                continue
            pos = row.get("position", [])
            if not isinstance(pos, list) or len(pos) != 2:
                continue
            r = int(pos[0])
            c = int(pos[1])
            x = c * self.tile_px
            y = r * self.tile_px
            sym = str(row.get("symbol", "M"))[:1] or "M"
            m = QGraphicsSimpleTextItem(sym)
            m.setBrush(QColor("#d66b6b"))
            m.setFont(QFont("DejaVu Sans Mono", 11, QFont.Weight.Bold))
            m.setPos(x + 5, y + 2)
            self.scene.addItem(m)

        # Animals
        for row in frame.get("animals", []):
            if not isinstance(row, dict) or not bool(row.get("alive", True)):
                continue
            pos = row.get("position", [])
            if not isinstance(pos, list) or len(pos) != 2:
                continue
            r = int(pos[0])
            c = int(pos[1])
            x = c * self.tile_px
            y = r * self.tile_px
            sym = str(row.get("symbol", "a"))[:1] or "a"
            a = QGraphicsSimpleTextItem(sym)
            a.setBrush(QColor("#9fe28a"))
            a.setFont(QFont("DejaVu Sans Mono", 11, QFont.Weight.Bold))
            a.setPos(x + 5, y + 2)
            self.scene.addItem(a)

        # Agents
        agents = frame.get("agents", {})
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
                faction_bg = self._faction_bg_color(faction_id)
                self.scene.addRect(x, y, self.tile_px, self.tile_px, QPen(Qt.PenStyle.NoPen), faction_bg)
                label = aid.split("_")[-1]
                t = QGraphicsSimpleTextItem(label)
                t.setBrush(QColor("#f3f7ff"))
                t.setFont(QFont("DejaVu Sans Mono", 11, QFont.Weight.Bold))
                t.setPos(x + 4, y + 2)
                self.scene.addItem(t)

        self.scene.setSceneRect(0, 0, max(1, w * self.tile_px), max(1, h * self.tile_px))
        self.frame_label.setText(f"frame {idx + 1}/{len(self.frames)}")

        agents_count = len(frame.get("agents", {})) if isinstance(frame.get("agents", {}), dict) else 0
        monsters_count = len([x for x in frame.get("monsters", []) if isinstance(x, dict) and bool(x.get("alive", True))])
        animals_count = len([x for x in frame.get("animals", []) if isinstance(x, dict) and bool(x.get("alive", True))])
        plants_count = len(frame.get("plant_plots", [])) if isinstance(frame.get("plant_plots", []), list) else 0
        resource_nodes_count = len(frame.get("resource_nodes", [])) if isinstance(frame.get("resource_nodes", []), list) else 0
        summary_lines = [
            f"size: {w}x{h}",
            f"agents: {agents_count}",
            f"monsters(alive): {monsters_count}",
            f"animals(alive): {animals_count}",
            f"plant_plots: {plants_count}",
            f"resource_nodes: {resource_nodes_count}",
        ]
        self.summary.setPlainText("\n".join(summary_lines))

        if idx > 0 and idx - 1 < len(self.step_logs):
            step = self.step_logs[idx - 1]
            lines: List[str] = []
            if isinstance(step.get("agents"), dict):
                for aid, row in sorted(step["agents"].items()):
                    if not isinstance(row, dict):
                        continue
                    action = row.get("action", -1)
                    reward = row.get("reward", 0.0)
                    events = row.get("events", [])
                    lines.append(f"{aid}: action={action} reward={reward:.3f}")
                    if isinstance(events, list) and events:
                        lines.extend([f"  - {e}" for e in events])
            self.log_text.setPlainText("\n".join(lines))
        else:
            self.log_text.setPlainText("(no step log for first frame)")


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
