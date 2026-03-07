#!/usr/bin/env python3
"""Convert a replay JSON file into an animated GIF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Set

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlrlgym.content.tiles import load_tileset  # noqa: E402

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception as exc:  # pragma: no cover - dependency/runtime specific
    raise RuntimeError("Pillow is required for GIF export. Install with: pip install pillow") from exc


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


def _color(name: str) -> str:
    return COLOR_MAP.get(str(name), "#d0d0d0")


def _parse_highlight(spec: str) -> tuple[bool, Set[str]]:
    raw = str(spec or "").strip().lower()
    if raw in ("", "none"):
        return False, set()
    if raw == "all":
        return True, set()
    return False, {part.strip() for part in raw.split(",") if part.strip()}


def _render_frame(
    frame: Dict[str, object],
    tile_defs: Dict[str, object],
    tile_px: int,
    highlight_all: bool,
    highlight_set: Set[str],
) -> Image.Image:
    grid = frame.get("grid", [])
    if not isinstance(grid, list) or not grid:
        raise ValueError("Frame is missing grid data")
    h = len(grid)
    w = len(grid[0]) if isinstance(grid[0], list) else 0
    if w <= 0:
        raise ValueError("Frame grid is empty")
    img = Image.new("RGB", (w * tile_px, h * tile_px), "#10151b")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    for r, row in enumerate(grid):
        if not isinstance(row, list):
            continue
        for c, tile_id_any in enumerate(row):
            tile_id = str(tile_id_any)
            td = tile_defs.get(tile_id)
            glyph = getattr(td, "glyph", "?")
            color_name = getattr(td, "color", "white")
            x0 = c * tile_px
            y0 = r * tile_px
            x1 = x0 + tile_px - 1
            y1 = y0 + tile_px - 1
            bg = "#1f232a" if (r + c) % 2 == 0 else "#242a33"
            draw.rectangle((x0, y0, x1, y1), fill=bg)
            draw.text((x0 + 5, y0 + 4), str(glyph)[:1], fill=_color(str(color_name)), font=font)

    agents = frame.get("agents", {})
    if isinstance(agents, dict):
        for aid, row in sorted(agents.items()):
            if not isinstance(row, dict) or not bool(row.get("alive", True)):
                continue
            pos = row.get("position", [])
            if not isinstance(pos, list) or len(pos) != 2:
                continue
            rr = int(pos[0])
            cc = int(pos[1])
            x0 = cc * tile_px
            y0 = rr * tile_px
            x1 = x0 + tile_px - 1
            y1 = y0 + tile_px - 1
            if highlight_all or aid in highlight_set:
                draw.rectangle((x0 + 1, y0 + 1, x1 - 1, y1 - 1), outline="#ffd166", width=2)
            draw.text((x0 + 4, y0 + 4), aid.split("_")[-1], fill="#64b2ff", font=font)

    for row in frame.get("monsters", []):
        if not isinstance(row, dict) or not bool(row.get("alive", True)):
            continue
        pos = row.get("position", [])
        if not isinstance(pos, list) or len(pos) != 2:
            continue
        rr = int(pos[0])
        cc = int(pos[1])
        draw.text((cc * tile_px + 4, rr * tile_px + 4), str(row.get("symbol", "M"))[:1], fill="#d66b6b", font=font)

    for row in frame.get("animals", []):
        if not isinstance(row, dict) or not bool(row.get("alive", True)):
            continue
        pos = row.get("position", [])
        if not isinstance(pos, list) or len(pos) != 2:
            continue
        rr = int(pos[0])
        cc = int(pos[1])
        draw.text((cc * tile_px + 4, rr * tile_px + 4), str(row.get("symbol", "a"))[:1], fill="#9fe28a", font=font)

    return img


def main() -> None:
    p = argparse.ArgumentParser(description="Convert *.replay.json to animated GIF")
    p.add_argument("replay_path", type=str, help="Path to replay JSON")
    p.add_argument("--output", type=str, default="", help="Output GIF path (default: <replay>.gif)")
    p.add_argument("--tiles-path", type=str, default="data/base/tiles.json")
    p.add_argument("--tile-size", type=int, default=24)
    p.add_argument("--fps", type=float, default=8.0)
    p.add_argument("--frames-per-step", type=int, default=1)
    p.add_argument(
        "--highlight-agents",
        type=str,
        default="none",
        help="Agent highlight mode: none, all, or comma-separated IDs (e.g. agent_0,agent_2)",
    )
    p.add_argument("--hold-start-seconds", type=float, default=1.0)
    p.add_argument("--hold-end-seconds", type=float, default=1.5)
    args = p.parse_args()

    replay_path = Path(args.replay_path).resolve()
    if not replay_path.exists():
        raise FileNotFoundError(replay_path)

    out_path = Path(args.output).resolve() if args.output else replay_path.with_suffix(".gif")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = json.loads(replay_path.read_text(encoding="utf-8"))
    frames_raw = payload.get("frames", [])
    if not isinstance(frames_raw, list) or not frames_raw:
        raise ValueError("Replay must contain a non-empty frames array")
    frames = [dict(row) for row in frames_raw if isinstance(row, dict)]
    if not frames:
        raise ValueError("Replay frames are empty after parsing")

    tile_defs = load_tileset(args.tiles_path)
    highlight_all, highlight_set = _parse_highlight(args.highlight_agents)
    fps = max(0.5, float(args.fps))
    frame_ms = int(round(1000.0 / fps))
    frames_per_step = max(1, int(args.frames_per_step))

    rendered: List[Image.Image] = []
    durations: List[int] = []
    for frame in frames:
        img = _render_frame(
            frame=frame,
            tile_defs=tile_defs,
            tile_px=max(8, int(args.tile_size)),
            highlight_all=highlight_all,
            highlight_set=highlight_set,
        )
        for _ in range(frames_per_step):
            rendered.append(img.copy())
            durations.append(frame_ms)

    hold_start = max(0, int(round(max(0.0, float(args.hold_start_seconds)) * fps)))
    hold_end = max(0, int(round(max(0.0, float(args.hold_end_seconds)) * fps)))
    if rendered:
        first = rendered[0]
        last = rendered[-1]
        for _ in range(hold_start):
            rendered.insert(0, first.copy())
            durations.insert(0, frame_ms)
        for _ in range(hold_end):
            rendered.append(last.copy())
            durations.append(frame_ms)

    rendered[0].save(
        out_path,
        save_all=True,
        append_images=rendered[1:],
        duration=durations,
        loop=0,
        optimize=False,
    )
    print(f"Wrote GIF: {out_path}")


if __name__ == "__main__":
    main()
