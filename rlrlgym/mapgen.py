"""Procedural map generation."""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .models import TileDef
from .tiles import weighted_tile_ids, weighted_tile_weights


def _weighted_choice(
    rng: random.Random, ids: List[str], cum_weights: List[float]
) -> str:
    pick = rng.random() * cum_weights[-1]
    idx = bisect.bisect_left(cum_weights, pick)
    if idx < 0:
        idx = 0
    if idx >= len(ids):
        idx = len(ids) - 1
    return ids[idx]


def _cum_weights(weights: List[float]) -> List[float]:
    out: List[float] = []
    total = 0.0
    for w in weights:
        total += max(0.0, float(w))
        out.append(total)
    return out


def _fade(t: float) -> float:
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _grad(hashv: int, x: float, y: float) -> float:
    h = hashv & 7
    u = x if h < 4 else y
    v = y if h < 4 else x
    return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)


def _build_permutation(rng: random.Random) -> List[int]:
    values = list(range(256))
    rng.shuffle(values)
    return values + values


def _perlin_2d(x: float, y: float, perm: List[int]) -> float:
    xi = int(math.floor(x)) & 255
    yi = int(math.floor(y)) & 255
    xf = x - math.floor(x)
    yf = y - math.floor(y)

    u = _fade(xf)
    v = _fade(yf)

    aa = perm[perm[xi] + yi]
    ab = perm[perm[xi] + yi + 1]
    ba = perm[perm[xi + 1] + yi]
    bb = perm[perm[xi + 1] + yi + 1]

    x1 = _lerp(_grad(aa, xf, yf), _grad(ba, xf - 1.0, yf), u)
    x2 = _lerp(_grad(ab, xf, yf - 1.0), _grad(bb, xf - 1.0, yf - 1.0), u)
    return _lerp(x1, x2, v)


def _fractal_noise_2d(
    x: float,
    y: float,
    perm: List[int],
    octaves: int,
    persistence: float = 0.5,
    lacunarity: float = 2.0,
) -> float:
    total = 0.0
    amplitude = 1.0
    frequency = 1.0
    norm = 0.0
    for _ in range(max(1, octaves)):
        total += amplitude * _perlin_2d(x * frequency, y * frequency, perm)
        norm += amplitude
        amplitude *= persistence
        frequency *= lacunarity
    if norm <= 0.0:
        return 0.5
    # Normalize Perlin output from [-1, 1] to [0, 1].
    return max(0.0, min(1.0, (total / norm) * 0.5 + 0.5))


def _valid_tile(tiles: Dict[str, TileDef], tile_id: str, fallback: str) -> str:
    if tile_id in tiles:
        return tile_id
    return fallback


@dataclass(slots=True)
class BiomeMap:
    patch_size: int
    patch_rows: int
    patch_cols: int
    field: List[List[str]]

    def biome_at_tile(self, r: int, c: int) -> str:
        pr = max(0, min(self.patch_rows - 1, r // self.patch_size))
        pc = max(0, min(self.patch_cols - 1, c // self.patch_size))
        return self.field[pr][pc]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patch_size": self.patch_size,
            "patch_rows": self.patch_rows,
            "patch_cols": self.patch_cols,
            "field": [list(row) for row in self.field],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "BiomeMap":
        return cls(
            patch_size=max(1, int(payload.get("patch_size", 1))),
            patch_rows=max(1, int(payload.get("patch_rows", 1))),
            patch_cols=max(1, int(payload.get("patch_cols", 1))),
            field=[list(row) for row in payload.get("field", [])],
        )


def generate_biome_map(
    rng: random.Random,
    biome_ids: List[str],
    biome_cum: List[float],
    patch_size: int,
    patch_rows: int,
    patch_cols: int,
    smooth_passes: int,
    noise_scale: float = 5.0,
    noise_octaves: int = 3,
    noise_mix: float = 0.2,
) -> BiomeMap:
    perm = _build_permutation(rng)
    xoff = rng.uniform(-1000.0, 1000.0)
    yoff = rng.uniform(-1000.0, 1000.0)
    total_w = biome_cum[-1]
    scale = max(1.0, float(noise_scale))
    mix = max(0.0, min(1.0, float(noise_mix)))

    field: List[List[str]] = []
    for pr in range(patch_rows):
        row: List[str] = []
        for pc in range(patch_cols):
            # Perlin drives broad contiguous regions; mix keeps local variety.
            noise = _fractal_noise_2d(
                (float(pc) / scale) + xoff,
                (float(pr) / scale) + yoff,
                perm=perm,
                octaves=noise_octaves,
            )
            if mix > 0.0:
                noise = ((1.0 - mix) * noise) + (mix * rng.random())
            pick = noise * total_w
            idx = bisect.bisect_left(biome_cum, pick)
            if idx < 0:
                idx = 0
            if idx >= len(biome_ids):
                idx = len(biome_ids) - 1
            row.append(biome_ids[idx])
        field.append(row)

    for _ in range(max(1, smooth_passes)):
        nxt = [list(row) for row in field]
        for pr in range(patch_rows):
            nr0 = max(0, pr - 1)
            nr1 = min(patch_rows, pr + 2)
            for pc in range(patch_cols):
                nc0 = max(0, pc - 1)
                nc1 = min(patch_cols, pc + 2)
                counts: Dict[str, int] = {}
                best_id = field[pr][pc]
                best_count = 0
                for nr in range(nr0, nr1):
                    row = field[nr]
                    for nc in range(nc0, nc1):
                        bid = row[nc]
                        count = counts.get(bid, 0) + 1
                        counts[bid] = count
                        if count > best_count or (count == best_count and bid < best_id):
                            best_id = bid
                            best_count = count
                nxt[pr][pc] = best_id
        field = nxt

    return BiomeMap(
        patch_size=patch_size,
        patch_rows=patch_rows,
        patch_cols=patch_cols,
        field=field,
    )


def _fill_disc(
    grid: List[List[str]],
    center_r: int,
    center_c: int,
    radius: int,
    tile_id: str,
    tile_id_edge: str,
) -> None:
    if not grid or not grid[0]:
        return
    h = len(grid)
    w = len(grid[0])
    r0 = max(1, center_r - radius)
    r1 = min(h - 2, center_r + radius)
    c0 = max(1, center_c - radius)
    c1 = min(w - 2, center_c + radius)
    rr = float(max(1, radius))
    for r in range(r0, r1 + 1):
        dr = float(r - center_r) / rr
        for c in range(c0, c1 + 1):
            dc = float(c - center_c) / rr
            dist = math.sqrt(dr * dr + dc * dc)
            if dist > 1.0:
                continue
            grid[r][c] = tile_id if dist <= 0.72 else tile_id_edge


def _fill_organic_blob(
    grid: List[List[str]],
    biomes: Dict[Tuple[int, int], str],
    center_r: int,
    center_c: int,
    radius: int,
    fill_tile_id: str,
    edge_tile_id: str,
    biome_id: str,
    perm: List[int],
    noise_scale: float,
    roughness: float,
) -> None:
    if not grid or not grid[0]:
        return
    h = len(grid)
    w = len(grid[0])
    r0 = max(1, center_r - radius)
    r1 = min(h - 2, center_r + radius)
    c0 = max(1, center_c - radius)
    c1 = min(w - 2, center_c + radius)
    rr = float(max(1, radius))
    scale = max(1.0, noise_scale)
    edge_band = 0.18
    for r in range(r0, r1 + 1):
        dr = float(r - center_r) / rr
        for c in range(c0, c1 + 1):
            dc = float(c - center_c) / rr
            dist = math.sqrt((dr * dr) + (dc * dc))
            if dist > 1.45:
                continue
            noise = _fractal_noise_2d(float(c) / scale, float(r) / scale, perm, octaves=3)
            radius_factor = 1.0 + ((noise - 0.5) * 2.0 * roughness)
            if dist > radius_factor:
                continue
            if dist > max(0.0, radius_factor - edge_band):
                grid[r][c] = edge_tile_id
            else:
                grid[r][c] = fill_tile_id
            biomes[(r, c)] = biome_id


def _paint_river(
    grid: List[List[str]],
    rng: random.Random,
    deep_tile: str,
    shallow_tile: str,
    min_width: int,
    max_width: int,
) -> None:
    if not grid or not grid[0]:
        return
    h = len(grid)
    w = len(grid[0])
    if h < 8 or w < 8:
        return
    from_left = bool(rng.randint(0, 1))
    if from_left:
        r = rng.randint(1, h - 2)
        c = 1
        end_c = w - 2
        step_c = 1
    else:
        r = rng.randint(1, h - 2)
        c = w - 2
        end_c = 1
        step_c = -1

    while c != end_c:
        width = rng.randint(min_width, max_width)
        for wc in range(-width, width + 1):
            rr = r + wc
            if rr <= 0 or rr >= h - 1:
                continue
            if abs(wc) <= max(0, width // 2):
                grid[rr][c] = deep_tile
            else:
                grid[rr][c] = shallow_tile
        c += step_c
        # Meander gently and occasionally wider bends.
        r += rng.choice((-1, 0, 0, 0, 1))
        r = max(1, min(h - 2, r))
        if rng.random() < 0.08:
            r += rng.choice((-1, 1))
            r = max(1, min(h - 2, r))


def generate_biome_terrain(
    width: int,
    height: int,
    tiles: Dict[str, TileDef],
    rng: random.Random,
    biome_defs: List[Dict[str, object]],
    wall_tile_id: str,
    floor_fallback_id: str,
    worldgen: Dict[str, object] | None = None,
    min_width: int = 4,
    min_height: int = 4,
) -> Tuple[List[List[str]], Dict[Tuple[int, int], str]]:
    worldgen = dict(worldgen or {})
    min_width = max(1, int(min_width))
    min_height = max(1, int(min_height))
    if width < min_width or height < min_height:
        raise ValueError(f"Map must be at least {min_width}x{min_height}")

    wall_id = _valid_tile(tiles, wall_tile_id, floor_fallback_id)
    floor_id = _valid_tile(tiles, floor_fallback_id, "floor")
    shallow_tile = _valid_tile(tiles, str(worldgen.get("shallow_water_tile_id", "shallow_water")), "water")
    deep_tile = _valid_tile(tiles, str(worldgen.get("deep_water_tile_id", "deep_water")), shallow_tile)
    tree_tile = _valid_tile(tiles, str(worldgen.get("tree_tile_id", "tree")), floor_id)
    stone_tile = _valid_tile(tiles, str(worldgen.get("stone_tile_id", "stone_wall")), wall_id)

    biome_ids: List[str] = []
    biome_weights: List[float] = []
    biome_tile_weight: Dict[str, Tuple[List[str], List[float], List[float]]] = {}
    global_tile_ids = weighted_tile_ids(tiles)
    global_tile_weights = weighted_tile_weights(tiles)
    global_cum = _cum_weights(global_tile_weights if global_tile_weights else [1.0])
    if not global_tile_ids:
        global_tile_ids = [floor_id]
    for row in biome_defs:
        bid = str(row.get("id", "")).strip()
        if not bid:
            continue
        biome_ids.append(bid)
        biome_weights.append(max(0.0, float(row.get("weight", 1.0))))
        raw = row.get("tile_weights", {})
        tile_ids: List[str] = []
        tile_weights: List[float] = []
        if isinstance(raw, dict):
            for tid, weight in raw.items():
                stid = str(tid)
                if stid not in tiles:
                    continue
                tile_ids.append(stid)
                tile_weights.append(max(0.0, float(weight)))
        if not tile_ids or sum(tile_weights) <= 0.0:
            tile_ids = list(global_tile_ids)
            tile_weights = list(global_tile_weights) if global_tile_weights else [1.0] * len(tile_ids)
        biome_tile_weight[bid] = (tile_ids, tile_weights, _cum_weights(tile_weights))

    if not biome_ids or sum(biome_weights) <= 0.0:
        raise ValueError("Biome definitions are required for biome terrain generation")
    biome_cum = _cum_weights(biome_weights)

    patch_size = max(8, int(worldgen.get("biome_patch_size", 24)))
    patch_rows = (height + patch_size - 1) // patch_size
    patch_cols = (width + patch_size - 1) // patch_size

    smooth_passes = max(1, int(worldgen.get("biome_smooth_passes", 2)))
    biome_noise_scale = max(1.0, float(worldgen.get("biome_noise_scale", 6.0)))
    biome_noise_octaves = max(1, int(worldgen.get("biome_noise_octaves", 3)))
    biome_noise_mix = max(0.0, min(1.0, float(worldgen.get("biome_noise_mix", 0.15))))
    biome_map = generate_biome_map(
        rng=rng,
        biome_ids=biome_ids,
        biome_cum=biome_cum,
        patch_size=patch_size,
        patch_rows=patch_rows,
        patch_cols=patch_cols,
        smooth_passes=smooth_passes,
        noise_scale=biome_noise_scale,
        noise_octaves=biome_noise_octaves,
        noise_mix=biome_noise_mix,
    )

    grid: List[List[str]] = [[floor_id for _ in range(width)] for _ in range(height)]
    biomes: Dict[Tuple[int, int], str] = {}

    for r in range(height):
        for c in range(width):
            if r in (0, height - 1) or c in (0, width - 1):
                grid[r][c] = wall_id
                continue
            bid = biome_map.biome_at_tile(r, c)
            biomes[(r, c)] = bid
            ids, _weights, cum = biome_tile_weight.get(bid, (global_tile_ids, global_tile_weights, global_cum))
            grid[r][c] = _weighted_choice(rng, ids, cum)

    # Lakes: deep core + shallow ring.
    lake_scale = max(1, int(worldgen.get("lake_scale", 170)))
    n_lakes = max(1, int((width * height) / float(lake_scale * lake_scale)))
    n_lakes = max(n_lakes, int(worldgen.get("min_lakes", 2)))
    n_lakes = min(n_lakes, int(worldgen.get("max_lakes", 24)))
    min_lake_r = max(3, int(worldgen.get("lake_min_radius", 6)))
    max_lake_r = max(min_lake_r, int(worldgen.get("lake_max_radius", 26)))
    organic_perm = _build_permutation(rng)
    for _ in range(n_lakes):
        cr = rng.randint(2 + max_lake_r, max(2 + max_lake_r, height - 3 - max_lake_r))
        cc = rng.randint(2 + max_lake_r, max(2 + max_lake_r, width - 3 - max_lake_r))
        rr = rng.randint(min_lake_r, max_lake_r)
        _fill_organic_blob(
            grid=grid,
            biomes=biomes,
            center_r=cr,
            center_c=cc,
            radius=rr,
            fill_tile_id=deep_tile,
            edge_tile_id=shallow_tile,
            biome_id="water",
            perm=organic_perm,
            noise_scale=float(worldgen.get("blob_noise_scale", 24.0)),
            roughness=float(worldgen.get("blob_noise_roughness", 0.35)),
        )

    # Rivers.
    min_rivers = max(1, int(worldgen.get("min_rivers", 2)))
    max_rivers = max(min_rivers, int(worldgen.get("max_rivers", 6)))
    n_rivers = rng.randint(min_rivers, max_rivers)
    river_w_min = max(1, int(worldgen.get("river_min_width", 2)))
    river_w_max = max(river_w_min, int(worldgen.get("river_max_width", 4)))
    for _ in range(n_rivers):
        _paint_river(grid, rng, deep_tile, shallow_tile, river_w_min, river_w_max)

    # Dense forest zones and exposed stone/mineral zones.
    area = width * height
    forest_blobs = max(2, int(area / float(max(1, int(worldgen.get("forest_cluster_scale", 130 * 130))))))
    stone_blobs = max(2, int(area / float(max(1, int(worldgen.get("stone_cluster_scale", 180 * 180))))))
    forest_radius = max(8, int(worldgen.get("forest_cluster_radius", 22)))
    stone_radius = max(8, int(worldgen.get("stone_cluster_radius", 18)))

    for _ in range(forest_blobs):
        cr = rng.randint(2 + forest_radius, max(2 + forest_radius, height - 3 - forest_radius))
        cc = rng.randint(2 + forest_radius, max(2 + forest_radius, width - 3 - forest_radius))
        _fill_organic_blob(
            grid=grid,
            biomes=biomes,
            center_r=cr,
            center_c=cc,
            radius=forest_radius,
            fill_tile_id=tree_tile,
            edge_tile_id="bush" if "bush" in tiles else tree_tile,
            biome_id="forest",
            perm=organic_perm,
            noise_scale=float(worldgen.get("blob_noise_scale", 24.0)),
            roughness=float(worldgen.get("blob_noise_roughness", 0.35)),
        )

    for _ in range(stone_blobs):
        cr = rng.randint(2 + stone_radius, max(2 + stone_radius, height - 3 - stone_radius))
        cc = rng.randint(2 + stone_radius, max(2 + stone_radius, width - 3 - stone_radius))
        _fill_organic_blob(
            grid=grid,
            biomes=biomes,
            center_r=cr,
            center_c=cc,
            radius=stone_radius,
            fill_tile_id=stone_tile,
            edge_tile_id=floor_id,
            biome_id="rocky",
            perm=organic_perm,
            noise_scale=float(worldgen.get("blob_noise_scale", 24.0)),
            roughness=float(worldgen.get("blob_noise_roughness", 0.35)),
        )

    fallback_tile = tiles.get(floor_id) or next(iter(tiles.values()))
    walkable = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if tiles.get(grid[r][c], fallback_tile).walkable
    ]
    if len(walkable) < 2:
        for r in range(1, height - 1):
            for c in range(1, width - 1):
                grid[r][c] = floor_id

    return grid, biomes


def generate_map(
    width: int,
    height: int,
    tiles: Dict[str, TileDef],
    rng: random.Random,
    wall_tile_id: str = "wall",
    floor_fallback_id: str = "floor",
    min_width: int = 4,
    min_height: int = 4,
) -> List[List[str]]:
    min_width = max(1, int(min_width))
    min_height = max(1, int(min_height))
    if width < min_width or height < min_height:
        raise ValueError(f"Map must be at least {min_width}x{min_height}")

    interior_ids = weighted_tile_ids(tiles)
    interior_weights = weighted_tile_weights(tiles)

    if not interior_ids:
        raise ValueError("No tiles available for map generation")

    grid: List[List[str]] = []
    for r in range(height):
        row: List[str] = []
        for c in range(width):
            if r in (0, height - 1) or c in (0, width - 1):
                row.append(wall_tile_id if wall_tile_id in tiles else floor_fallback_id)
            else:
                row.append(rng.choices(interior_ids, weights=interior_weights, k=1)[0])
        grid.append(row)

    # Ensure enough walkable floor to place agents.
    walkable = [
        (r, c)
        for r in range(height)
        for c in range(width)
        if tiles[grid[r][c]].walkable
    ]
    if len(walkable) < 2:
        for r in range(1, height - 1):
            for c in range(1, width - 1):
                grid[r][c] = floor_fallback_id

    return grid


def sample_walkable_positions(
    grid: List[List[str]], tiles: Dict[str, TileDef], count: int, rng: random.Random
) -> List[Tuple[int, int]]:
    walkable = [
        (r, c)
        for r, row in enumerate(grid)
        for c, tile_id in enumerate(row)
        if tiles[tile_id].walkable
    ]
    if len(walkable) < count:
        raise ValueError("Not enough walkable cells for agent placement")
    rng.shuffle(walkable)
    return walkable[:count]
