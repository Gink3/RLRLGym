#!/usr/bin/env python3
"""PyQt6 map builder: generate, inspect, and manually save static maps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Dict, List, Tuple

from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QToolTip,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlrlgym import EnvConfig, PettingZooParallelRLRLGym  # noqa: E402
from rlrlgym.models import TileDef  # noqa: E402
from rlrlgym.tiles import load_tileset  # noqa: E402


TILE_SIZE = 32


def _map_payload(
    *,
    name: str,
    seed: int,
    grid: List[List[str]],
    biomes: Dict[Tuple[int, int], str],
) -> Dict[str, object]:
    biome_rows = [
        {"position": [r, c], "biome": biome}
        for (r, c), biome in sorted(biomes.items())
        if str(biome).strip()
    ]
    return {
        "schema_version": 1,
        "map": {
            "name": str(name),
            "seed": int(seed),
            "width": int(len(grid[0]) if grid else 0),
            "height": int(len(grid)),
            "grid": [[str(tile) for tile in row] for row in grid],
            "biomes": biome_rows,
        },
    }


class TileSprites:
    def __init__(self, tile_defs: Dict[str, TileDef], size: int = TILE_SIZE) -> None:
        self.tile_defs = tile_defs
        self.size = int(size)
        self.cache: Dict[str, QPixmap] = {}

    def update_defs(self, tile_defs: Dict[str, TileDef]) -> None:
        self.tile_defs = tile_defs
        self.cache.clear()

    def pixmap(self, tile_id: str) -> QPixmap:
        if tile_id in self.cache:
            return self.cache[tile_id]
        pm = QPixmap(self.size, self.size)
        pm.fill(QColor("#2a3038"))
        p = QPainter(pm)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        # Base palettes.
        if tile_id in {"water", "deep_water"}:
            self._paint_water(p, deep=True)
        elif tile_id == "shallow_water":
            self._paint_water(p, deep=False)
        elif tile_id in {"grass", "berry_plant", "grain_plant", "herb_plant"}:
            self._paint_grass(p)
            if tile_id == "berry_plant":
                self._dot(p, QColor("#d082d6"), 10, 10, 3)
                self._dot(p, QColor("#b65cbf"), 18, 16, 3)
            elif tile_id == "grain_plant":
                p.setPen(QPen(QColor("#d4b24a"), 2))
                p.drawLine(10, 22, 10, 8)
                p.drawLine(16, 22, 16, 7)
                p.drawLine(22, 22, 22, 9)
            elif tile_id == "herb_plant":
                p.setPen(QPen(QColor("#67b35c"), 2))
                p.drawLine(16, 23, 12, 12)
                p.drawLine(16, 23, 20, 11)
        elif tile_id == "bush":
            self._paint_grass(p)
            p.setBrush(QColor("#3e8c4a"))
            p.setPen(QPen(QColor("#2b6d38"), 1))
            p.drawEllipse(4, 11, 11, 11)
            p.drawEllipse(11, 8, 13, 13)
            p.drawEllipse(18, 12, 10, 10)
        elif tile_id == "tree":
            self._paint_grass(p)
            p.setBrush(QColor("#4a3120"))
            p.setPen(Qt.PenStyle.NoPen)
            p.drawRect(13, 16, 6, 12)
            p.setBrush(QColor("#2f8b45"))
            p.drawEllipse(5, 5, 22, 16)
        elif tile_id in {"wall", "stone_wall", "wood_wall"}:
            if tile_id == "wood_wall":
                self._paint_wood_wall(p)
            else:
                self._paint_stone_wall(p)
        elif tile_id == "stone_floor":
            self._paint_stone_floor(p)
        elif tile_id == "shrine":
            self._paint_stone_floor(p)
            p.setPen(QPen(QColor("#d08bd6"), 2))
            p.drawLine(16, 6, 16, 24)
            p.drawLine(10, 10, 22, 10)
            p.drawLine(12, 18, 20, 18)
        elif tile_id == "chest":
            self._paint_grass(p)
            p.setBrush(QColor("#9b6b2f"))
            p.setPen(QPen(QColor("#6d471c"), 1))
            p.drawRect(6, 11, 20, 14)
            p.setBrush(QColor("#d4b24a"))
            p.drawRect(14, 11, 4, 14)
        else:
            # Default/fallback tile sprite for unknown tile IDs.
            td = self.tile_defs.get(tile_id)
            fg = QColor("#d0d0d0") if td is None else QColor("#d0d0d0")
            p.fillRect(0, 0, self.size, self.size, QColor("#2d333d"))
            p.setPen(QPen(fg, 2))
            p.drawRect(3, 3, self.size - 6, self.size - 6)
            p.drawLine(6, 6, self.size - 6, self.size - 6)
            p.drawLine(self.size - 6, 6, 6, self.size - 6)

        p.end()
        self.cache[tile_id] = pm
        return pm

    def _dot(self, p: QPainter, color: QColor, x: int, y: int, r: int) -> None:
        p.setBrush(color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(x, y, r, r)

    def _paint_grass(self, p: QPainter) -> None:
        p.fillRect(0, 0, self.size, self.size, QColor("#29442d"))
        p.setPen(QPen(QColor("#3f6f45"), 1))
        for x in range(3, self.size, 6):
            p.drawLine(x, self.size, x + 2, self.size - 8)

    def _paint_stone_floor(self, p: QPainter) -> None:
        p.fillRect(0, 0, self.size, self.size, QColor("#3a3f47"))
        p.setPen(QPen(QColor("#4d535d"), 1))
        for y in range(4, self.size, 8):
            p.drawLine(2, y, self.size - 2, y)
        for x in range(5, self.size, 9):
            p.drawLine(x, 2, x, self.size - 2)

    def _paint_stone_wall(self, p: QPainter) -> None:
        p.fillRect(0, 0, self.size, self.size, QColor("#2f343d"))
        p.setPen(QPen(QColor("#575e69"), 2))
        for y in (7, 16, 25):
            p.drawLine(2, y, self.size - 2, y)
        for x in (9, 19):
            p.drawLine(x, 2, x, self.size - 2)

    def _paint_wood_wall(self, p: QPainter) -> None:
        p.fillRect(0, 0, self.size, self.size, QColor("#5a3f29"))
        p.setPen(QPen(QColor("#7a5538"), 2))
        for x in (6, 13, 20, 27):
            p.drawLine(x, 2, x, self.size - 2)

    def _paint_water(self, p: QPainter, deep: bool) -> None:
        p.fillRect(0, 0, self.size, self.size, QColor("#2a4a82" if deep else "#3f73b8"))
        p.setPen(QPen(QColor("#7fb4e0" if deep else "#a8d4ee"), 2))
        p.drawArc(2, 9, 11, 7, 0, 180 * 16)
        p.drawArc(10, 17, 13, 7, 0, 180 * 16)
        p.drawArc(18, 9, 11, 7, 0, 180 * 16)


class MapCanvas(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.grid: List[List[str]] = []
        self.biomes: Dict[Tuple[int, int], str] = {}
        self.tile_defs: Dict[str, TileDef] = {}
        self.sprites = TileSprites(self.tile_defs, TILE_SIZE)

        self.zoom = 1.0
        self.min_zoom = 0.25
        self.max_zoom = 4.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.dragging = False
        self.last_mouse = QPoint(0, 0)
        self.hover_cb = None

    def set_hover_callback(self, cb) -> None:
        self.hover_cb = cb

    def set_map(
        self,
        grid: List[List[str]],
        biomes: Dict[Tuple[int, int], str],
        tile_defs: Dict[str, TileDef],
    ) -> None:
        self.grid = grid
        self.biomes = biomes
        self.tile_defs = tile_defs
        self.sprites.update_defs(tile_defs)
        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.update()

    def _tile_screen_size(self) -> float:
        return float(TILE_SIZE) * float(self.zoom)

    def _screen_to_tile(self, x: float, y: float) -> Tuple[int, int]:
        ts = self._tile_screen_size()
        if ts <= 0.0:
            return -1, -1
        c = int((x - self.offset_x) // ts)
        r = int((y - self.offset_y) // ts)
        return r, c

    def paintEvent(self, event) -> None:  # type: ignore[override]
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#151a20"))
        if not self.grid or not self.grid[0]:
            p.end()
            return

        h = len(self.grid)
        w = len(self.grid[0])
        ts = self._tile_screen_size()

        c0 = max(0, int((-self.offset_x) // ts) - 1)
        r0 = max(0, int((-self.offset_y) // ts) - 1)
        c1 = min(w - 1, int((self.width() - self.offset_x) // ts) + 1)
        r1 = min(h - 1, int((self.height() - self.offset_y) // ts) + 1)

        if c1 < c0 or r1 < r0:
            p.end()
            return

        for r in range(r0, r1 + 1):
            row = self.grid[r]
            sy = self.offset_y + (r * ts)
            for c in range(c0, c1 + 1):
                tile_id = str(row[c])
                sx = self.offset_x + (c * ts)
                pm = self.sprites.pixmap(tile_id)
                p.drawPixmap(int(sx), int(sy), int(ts), int(ts), pm)

        p.end()

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.0015 ** delta
        old_zoom = self.zoom
        new_zoom = max(self.min_zoom, min(self.max_zoom, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        mx = float(event.position().x())
        my = float(event.position().y())
        world_x = (mx - self.offset_x) / (float(TILE_SIZE) * old_zoom)
        world_y = (my - self.offset_y) / (float(TILE_SIZE) * old_zoom)
        self.zoom = new_zoom
        self.offset_x = mx - (world_x * float(TILE_SIZE) * self.zoom)
        self.offset_y = my - (world_y * float(TILE_SIZE) * self.zoom)
        self.update()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = True
            self.last_mouse = event.pos()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self.dragging:
            d = event.pos() - self.last_mouse
            self.last_mouse = event.pos()
            self.offset_x += float(d.x())
            self.offset_y += float(d.y())
            self.update()

        if self.hover_cb is not None:
            r, c = self._screen_to_tile(float(event.position().x()), float(event.position().y()))
            self.hover_cb(r, c, event.globalPosition().toPoint())


class MapBuilderWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__()
        self.setWindowTitle("RLRLGym Map Builder (PyQt6)")
        self.resize(1600, 950)

        self.args = args
        self.generated_grid: List[List[str]] = []
        self.generated_biomes: Dict[Tuple[int, int], str] = {}
        self.last_seed: int | None = None
        self.tile_defs = load_tileset(str(args.tiles_path))

        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QHBoxLayout(root)

        left = QVBoxLayout()
        layout.addLayout(left, stretch=5)

        controls = QFormLayout()
        left.addLayout(controls)

        self.name_edit = QLineEdit(str(args.name))
        self.seed_edit = QLineEdit(str(args.seed))
        self.lock_seed_check = QCheckBox("Use fixed seed")
        self.lock_seed_check.setChecked(False)
        self.width_edit = QLineEdit(str(args.width))
        self.height_edit = QLineEdit(str(args.height))
        self.tiles_path_edit = QLineEdit(str(args.tiles_path))
        self.mapgen_path_edit = QLineEdit(str(args.mapgen_config_path))

        tile_path_row = QHBoxLayout()
        tile_path_row.addWidget(self.tiles_path_edit)
        self.browse_tiles_btn = QPushButton("Browse")
        tile_path_row.addWidget(self.browse_tiles_btn)

        mapgen_path_row = QHBoxLayout()
        mapgen_path_row.addWidget(self.mapgen_path_edit)
        self.browse_mapgen_btn = QPushButton("Browse")
        mapgen_path_row.addWidget(self.browse_mapgen_btn)

        controls.addRow("Map Name", self.name_edit)
        controls.addRow("Seed", self.seed_edit)
        controls.addRow("", self.lock_seed_check)
        controls.addRow("Width", self.width_edit)
        controls.addRow("Height", self.height_edit)
        controls.addRow("Tiles JSON", tile_path_row)
        controls.addRow("Mapgen JSON", mapgen_path_row)

        btns = QHBoxLayout()
        self.generate_btn = QPushButton("Generate Map")
        self.save_btn = QPushButton("Save Map As...")
        self.save_btn.setEnabled(False)
        btns.addWidget(self.generate_btn)
        btns.addWidget(self.save_btn)
        left.addLayout(btns)

        self.canvas = MapCanvas(self)
        self.canvas.set_hover_callback(self._on_hover)
        left.addWidget(self.canvas, stretch=1)

        right = QVBoxLayout()
        layout.addLayout(right, stretch=2)

        right.addWidget(QLabel("Map Summary"))
        self.summary = QTextEdit(self)
        self.summary.setReadOnly(True)
        right.addWidget(self.summary, stretch=1)

        right.addWidget(QLabel("Tile Legend"))
        self.legend = QListWidget(self)
        right.addWidget(self.legend, stretch=1)

        right.addWidget(QLabel("Hover Info"))
        self.hover_info = QTextEdit(self)
        self.hover_info.setReadOnly(True)
        right.addWidget(self.hover_info, stretch=1)

        self.browse_tiles_btn.clicked.connect(self._pick_tiles)
        self.browse_mapgen_btn.clicked.connect(self._pick_mapgen)
        self.generate_btn.clicked.connect(self._generate_map)
        self.save_btn.clicked.connect(self._save_map)

        self._apply_dark_theme()
        self._generate_map()

    def _apply_dark_theme(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow { background: #11161c; color: #dbe6f3; }
            QWidget { background: #1b232d; color: #dbe6f3; }
            QLineEdit, QTextEdit, QListWidget {
                background: #10161d;
                color: #dbe6f3;
                border: 1px solid #2d3a47;
                border-radius: 4px;
                padding: 4px;
            }
            QLabel { color: #cfe0f0; }
            QPushButton {
                background: #243140;
                color: #e5eef7;
                border: 1px solid #3a4a5e;
                border-radius: 4px;
                padding: 6px 10px;
            }
            QPushButton:hover { background: #2d3e52; }
            QPushButton:disabled { background: #1a2531; color: #8092a5; }
            QToolTip {
                background: #0f141a;
                color: #e6f0fa;
                border: 1px solid #3a4a5e;
            }
            """
        )

    def _pick_tiles(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select tiles JSON",
            str(Path("data/base")),
            "JSON (*.json)",
        )
        if p:
            self.tiles_path_edit.setText(str(Path(p)))

    def _pick_mapgen(self) -> None:
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select mapgen JSON",
            str(Path("data/base")),
            "JSON (*.json)",
        )
        if p:
            self.mapgen_path_edit.setText(str(Path(p)))

    def _generate_map(self) -> None:
        try:
            width = max(4, int(self.width_edit.text().strip() or "80"))
            height = max(4, int(self.height_edit.text().strip() or "80"))
            if self.lock_seed_check.isChecked():
                seed = int(self.seed_edit.text().strip() or "7")
            else:
                seed = random.SystemRandom().randint(1, 2_147_483_647)
                self.seed_edit.setText(str(seed))
            tiles_path = str(self.tiles_path_edit.text().strip() or "data/base/tiles.json")
            mapgen_path = str(self.mapgen_path_edit.text().strip() or "data/base/mapgen_config.json")

            self.tile_defs = load_tileset(tiles_path)

            cfg = EnvConfig(
                width=width,
                height=height,
                max_steps=1,
                n_agents=2,
                render_enabled=False,
                tiles_path=tiles_path,
                mapgen_config_path=mapgen_path,
            )
            env = PettingZooParallelRLRLGym(cfg)
            env.reset(seed=seed)
            assert env.state is not None

            self.generated_grid = [[str(tile) for tile in row] for row in env.state.grid]
            self.generated_biomes = dict(env.state.biomes)
            self.last_seed = seed
            self.canvas.set_map(self.generated_grid, self.generated_biomes, self.tile_defs)
            self.save_btn.setEnabled(True)
            self._update_summary(seed)
            self._update_legend()
        except Exception as exc:
            QMessageBox.critical(self, "Map generation failed", str(exc))

    def _update_summary(self, seed: int) -> None:
        grid = self.generated_grid
        if not grid:
            self.summary.setPlainText("No map generated.")
            return
        h = len(grid)
        w = len(grid[0]) if h > 0 else 0

        counts: Dict[str, int] = {}
        for row in grid:
            for tid in row:
                counts[tid] = counts.get(tid, 0) + 1

        lines = [
            f"name={self.name_edit.text().strip() or 'generated_map'}",
            f"seed={seed}",
            f"size={w}x{h}",
            f"biome_cells={len(self.generated_biomes)}",
            "",
            "tile_count_total=" + str(sum(counts.values())),
        ]
        self.summary.setPlainText("\n".join(lines))

    def _update_legend(self) -> None:
        self.legend.clear()
        if not self.generated_grid:
            return
        counts: Dict[str, int] = {}
        for row in self.generated_grid:
            for tid in row:
                counts[tid] = counts.get(tid, 0) + 1
        for tid, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            td = self.tile_defs.get(tid)
            walkable = bool(td.walkable) if td is not None else False
            pm = self.canvas.sprites.pixmap(tid)
            item = QListWidgetItem(QIcon(pm), f"{tid}  count={cnt}  walkable={walkable}")
            self.legend.addItem(item)

    def _on_hover(self, r: int, c: int, global_pos: QPoint) -> None:
        grid = self.generated_grid
        if not grid or not grid[0]:
            return
        if r < 0 or c < 0 or r >= len(grid) or c >= len(grid[0]):
            self.hover_info.setPlainText("outside map")
            return
        tid = str(grid[r][c])
        td = self.tile_defs.get(tid)
        biome = self.generated_biomes.get((r, c), "")
        walkable = bool(td.walkable) if td is not None else False
        info = (
            f"position=({r}, {c})\n"
            f"tile_id={tid}\n"
            f"walkable={walkable}\n"
            f"biome={biome if biome else '<none>'}"
        )
        self.hover_info.setPlainText(info)
        QToolTip.showText(global_pos, info, self.canvas)

    def _save_map(self) -> None:
        if not self.generated_grid:
            QMessageBox.warning(self, "Save map", "Generate a map first.")
            return
        name = self.name_edit.text().strip() or "generated_map"
        seed = self.last_seed if self.last_seed is not None else int(self.seed_edit.text().strip() or "7")
        payload = _map_payload(
            name=name,
            seed=seed,
            grid=self.generated_grid,
            biomes=self.generated_biomes,
        )
        out, _ = QFileDialog.getSaveFileName(
            self,
            "Save map JSON",
            str(Path("data/maps") / f"{name}.json"),
            "JSON (*.json)",
        )
        if not out:
            return
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        QMessageBox.information(self, "Map saved", f"Saved: {out_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Open Map Builder (PyQt6)")
    p.add_argument("--width", type=int, default=80)
    p.add_argument("--height", type=int, default=80)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--name", type=str, default="generated_map")
    p.add_argument("--tiles-path", type=str, default="data/base/tiles.json")
    p.add_argument("--mapgen-config-path", type=str, default="data/base/mapgen_config.json")
    return p


def main() -> None:
    args = build_parser().parse_args()
    app = QApplication(sys.argv)
    win = MapBuilderWindow(args)
    win.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
