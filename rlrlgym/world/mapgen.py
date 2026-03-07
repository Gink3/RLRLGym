"""Procedural map generation."""

from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from ..systems.models import TileDef
from ..content.tiles import weighted_tile_ids, weighted_tile_weights

OPEN_WATER_TILE_IDS = {"water", "shallow_water", "deep_water"}

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
    biomes: Dict[Tuple[int, int], str],
    rng: random.Random,
    deep_tile: str,
    shallow_tile: str,
    min_width: int,
    max_width: int,
    *,
    cross_map_probability: float = 0.7,
    endpoint_margin: int = 6,
    meander_strength: float = 1.2,
) -> None:
    if not grid or not grid[0]:
        return
    h = len(grid)
    w = len(grid[0])
    if h < 8 or w < 8:
        return
    endpoint_margin = max(2, int(endpoint_margin))
    meander_strength = max(0.0, float(meander_strength))
    cross_map_probability = max(0.0, min(1.0, float(cross_map_probability)))

    def _interior_point() -> Tuple[int, int]:
        r_lo = max(1, endpoint_margin)
        r_hi = min(h - 2, h - 1 - endpoint_margin)
        c_lo = max(1, endpoint_margin)
        c_hi = min(w - 2, w - 1 - endpoint_margin)
        if r_lo > r_hi:
            r_lo, r_hi = 1, h - 2
        if c_lo > c_hi:
            c_lo, c_hi = 1, w - 2
        return rng.randint(r_lo, r_hi), rng.randint(c_lo, c_hi)

    def _carve(r: int, c: int, width: int) -> None:
        if r <= 0 or r >= h - 1 or c <= 0 or c >= w - 1:
            return
        radius = max(1, int(width))
        for dr in range(-radius, radius + 1):
            rr = r + dr
            if rr <= 0 or rr >= h - 1:
                continue
            for dc in range(-radius, radius + 1):
                cc = c + dc
                if cc <= 0 or cc >= w - 1:
                    continue
                dist = math.sqrt(float((dr * dr) + (dc * dc)))
                if dist > float(radius):
                    continue
                grid[rr][cc] = deep_tile if dist <= max(1.0, float(radius) * 0.6) else shallow_tile
                biomes[(rr, cc)] = "water"

    river_is_cross_map = rng.random() < cross_map_probability
    if river_is_cross_map:
        horizontal = bool(rng.randint(0, 1))
        if horizontal:
            start_r = rng.randint(1, h - 2)
            start_c = 1 if bool(rng.randint(0, 1)) else w - 2
            end_r = rng.randint(1, h - 2)
            end_c = w - 2 if start_c == 1 else 1
        else:
            start_r = 1 if bool(rng.randint(0, 1)) else h - 2
            start_c = rng.randint(1, w - 2)
            end_r = h - 2 if start_r == 1 else 1
            end_c = rng.randint(1, w - 2)
    else:
        start_r, start_c = _interior_point()
        end_r, end_c = _interior_point()
        min_sep = max(8, min((h + w) // 6, 20))
        retries = 0
        while (
            abs(start_r - end_r) + abs(start_c - end_c) < min_sep
            and retries < 20
        ):
            end_r, end_c = _interior_point()
            retries += 1

    if river_is_cross_map:
        row_min = 1
        row_max = h - 2
        col_min = 1
        col_max = w - 2
    else:
        row_min = max(1, endpoint_margin)
        row_max = min(h - 2, h - 1 - endpoint_margin)
        col_min = max(1, endpoint_margin)
        col_max = min(w - 2, w - 1 - endpoint_margin)
        if row_min > row_max:
            row_min, row_max = 1, h - 2
        if col_min > col_max:
            col_min, col_max = 1, w - 2

    cur_r = start_r
    cur_c = start_c
    max_steps = max(24, (h + w) * 4)
    steps = 0
    while steps < max_steps:
        width = rng.randint(min_width, max_width)
        _carve(cur_r, cur_c, width)
        if cur_r == end_r and cur_c == end_c:
            break
        best: Tuple[int, int] | None = None
        best_score = float("inf")
        candidates = [(cur_r - 1, cur_c), (cur_r + 1, cur_c), (cur_r, cur_c - 1), (cur_r, cur_c + 1)]
        rng.shuffle(candidates)
        for nr, nc in candidates:
            if nr < row_min or nr > row_max or nc < col_min or nc > col_max:
                continue
            target = abs(nr - end_r) + abs(nc - end_c)
            score = float(target) + (rng.random() * meander_strength)
            if score < best_score:
                best_score = score
                best = (nr, nc)
        if best is None:
            break
        cur_r, cur_c = best
        # Occasional meander kick to avoid straight-line barriers.
        if rng.random() < 0.08:
            cur_r = max(row_min, min(row_max, cur_r + rng.choice((-1, 1))))
            cur_c = max(col_min, min(col_max, cur_c + rng.choice((-1, 1))))
        steps += 1


def _can_place_rect(
    grid: List[List[str]],
    top: int,
    left: int,
    rect_h: int,
    rect_w: int,
    blocked_tiles: set[str],
) -> bool:
    h = len(grid)
    w = len(grid[0]) if h > 0 else 0
    if top <= 1 or left <= 1 or top + rect_h >= h - 1 or left + rect_w >= w - 1:
        return False
    for r in range(top - 1, top + rect_h + 1):
        for c in range(left - 1, left + rect_w + 1):
            if grid[r][c] in blocked_tiles:
                return False
    return True


def _paint_ruin(
    grid: List[List[str]],
    biomes: Dict[Tuple[int, int], str],
    rng: random.Random,
    *,
    top: int,
    left: int,
    rect_h: int,
    rect_w: int,
    wall_tile: str,
    floor_tile: str,
    biome_id: str,
) -> None:
    bottom = top + rect_h - 1
    right = left + rect_w - 1
    door_side = rng.choice(("n", "s", "e", "w"))

    for r in range(top, bottom + 1):
        for c in range(left, right + 1):
            is_border = r in (top, bottom) or c in (left, right)
            if is_border:
                carve = False
                if door_side == "n" and r == top and c == left + (rect_w // 2):
                    carve = True
                elif door_side == "s" and r == bottom and c == left + (rect_w // 2):
                    carve = True
                elif door_side == "w" and c == left and r == top + (rect_h // 2):
                    carve = True
                elif door_side == "e" and c == right and r == top + (rect_h // 2):
                    carve = True
                elif rng.random() < 0.2:
                    carve = True
                grid[r][c] = floor_tile if carve else wall_tile
            else:
                grid[r][c] = floor_tile
            biomes[(r, c)] = biome_id


def _place_structures(
    grid: List[List[str]],
    biomes: Dict[Tuple[int, int], str],
    tiles: Dict[str, TileDef],
    rng: random.Random,
    structures: List[Dict[str, object]],
    floor_fallback_id: str,
) -> None:
    if not structures:
        return
    height = len(grid)
    width = len(grid[0]) if height > 0 else 0
    if width < 8 or height < 8:
        return

    blocked_tiles = {"deep_water", "shallow_water", "water", "indestructible_wall"}
    for row in structures:
        kind = str(row.get("id", "ruins")).strip().lower()
        if kind not in {"ruins", "ruin"}:
            continue
        density = max(0.0, float(row.get("density", 0.004)))
        min_count = max(0, int(row.get("min_count", 0)))
        max_count = max(min_count, int(row.get("max_count", 999999)))
        est = max(min_count, int((width * height) * density))
        target = min(max_count, est)
        if target <= 0:
            continue

        min_w = max(5, int(row.get("min_width", 6)))
        max_w = max(min_w, int(row.get("max_width", 12)))
        min_h = max(5, int(row.get("min_height", 6)))
        max_h = max(min_h, int(row.get("max_height", 12)))
        wall_tile = _valid_tile(tiles, str(row.get("wall_tile_id", "rock_wall")), "rock_wall")
        floor_tile = _valid_tile(tiles, str(row.get("floor_tile_id", floor_fallback_id)), floor_fallback_id)
        biome_id = str(row.get("biome_id", "ruins")).strip() or "ruins"

        placed = 0
        attempts = 0
        max_attempts = max(40, target * 60)
        while placed < target and attempts < max_attempts:
            attempts += 1
            rect_w = rng.randint(min_w, max_w)
            rect_h = rng.randint(min_h, max_h)
            if rect_w >= width - 4 or rect_h >= height - 4:
                continue
            left = rng.randint(2, width - rect_w - 3)
            top = rng.randint(2, height - rect_h - 3)
            if not _can_place_rect(grid, top, left, rect_h, rect_w, blocked_tiles):
                continue
            _paint_ruin(
                grid,
                biomes,
                rng,
                top=top,
                left=left,
                rect_h=rect_h,
                rect_w=rect_w,
                wall_tile=wall_tile,
                floor_tile=floor_tile,
                biome_id=biome_id,
            )
            placed += 1


def _gaussian_kernel_1d(sigma: float, radius: int) -> List[float]:
    radius = max(1, int(radius))
    sigma = max(0.5, float(sigma))
    vals: List[float] = []
    denom = 2.0 * sigma * sigma
    for i in range(-radius, radius + 1):
        vals.append(math.exp(-(float(i * i)) / denom))
    total = sum(vals)
    if total <= 0.0:
        return [1.0 / float((radius * 2) + 1)] * ((radius * 2) + 1)
    return [v / total for v in vals]


def _blur_mask(mask: List[List[float]], sigma: float, radius: int) -> List[List[float]]:
    if not mask or not mask[0]:
        return mask
    h = len(mask)
    w = len(mask[0])
    kernel = _gaussian_kernel_1d(sigma=sigma, radius=radius)
    kr = len(kernel) // 2

    # Horizontal pass.
    tmp: List[List[float]] = [[0.0 for _ in range(w)] for _ in range(h)]
    for r in range(h):
        row = mask[r]
        out_row = tmp[r]
        for c in range(w):
            s = 0.0
            for ki, kv in enumerate(kernel):
                cc = c + (ki - kr)
                if cc < 0:
                    cc = 0
                elif cc >= w:
                    cc = w - 1
                s += row[cc] * kv
            out_row[c] = s

    # Vertical pass.
    out: List[List[float]] = [[0.0 for _ in range(w)] for _ in range(h)]
    for r in range(h):
        out_row = out[r]
        for c in range(w):
            s = 0.0
            for ki, kv in enumerate(kernel):
                rr = r + (ki - kr)
                if rr < 0:
                    rr = 0
                elif rr >= h:
                    rr = h - 1
                s += tmp[rr][c] * kv
            out_row[c] = s
    return out


def _apply_forest_density_mask(
    grid: List[List[str]],
    biomes: Dict[Tuple[int, int], str],
    rng: random.Random,
    *,
    forest_blobs: int,
    forest_radius: int,
    tree_tile: str,
    bush_tile: str,
    fallback_floor: str,
    worldgen: Dict[str, object],
) -> None:
    if not grid or not grid[0] or forest_blobs <= 0:
        return
    h = len(grid)
    w = len(grid[0])
    if h < 6 or w < 6:
        return

    mask: List[List[float]] = [[0.0 for _ in range(w)] for _ in range(h)]
    perm = _build_permutation(rng)
    warp_scale = max(8.0, float(worldgen.get("forest_warp_scale", 26.0)))
    warp_strength = max(0.0, min(0.5, float(worldgen.get("forest_warp_strength", 0.15))))

    for _ in range(forest_blobs):
        cr = rng.randint(2 + forest_radius, max(2 + forest_radius, h - 3 - forest_radius))
        cc = rng.randint(2 + forest_radius, max(2 + forest_radius, w - 3 - forest_radius))
        rr = float(max(1, forest_radius))
        r0 = max(1, cr - forest_radius)
        r1 = min(h - 2, cr + forest_radius)
        c0 = max(1, cc - forest_radius)
        c1 = min(w - 2, cc + forest_radius)
        for r in range(r0, r1 + 1):
            dr = float(r - cr) / rr
            for c in range(c0, c1 + 1):
                dc = float(c - cc) / rr
                d = math.sqrt((dr * dr) + (dc * dc))
                if d > 1.1:
                    continue
                base = max(0.0, 1.0 - d)
                if base <= 0.0:
                    continue
                n = _fractal_noise_2d(float(c) / warp_scale, float(r) / warp_scale, perm, octaves=3)
                warped = max(0.0, min(1.0, base + ((n - 0.5) * 2.0 * warp_strength)))
                if warped > mask[r][c]:
                    mask[r][c] = warped

    sigma = max(0.6, float(worldgen.get("forest_blur_sigma", 1.9)))
    blur_radius = max(1, int(worldgen.get("forest_blur_radius", 3)))
    mask = _blur_mask(mask, sigma=sigma, radius=blur_radius)

    tree_threshold = max(0.30, min(0.95, float(worldgen.get("forest_tree_threshold", 0.62))))
    bush_threshold = max(0.08, min(tree_threshold - 0.02, float(worldgen.get("forest_bush_threshold", 0.32))))
    max_tree_density = max(0.1, min(0.95, float(worldgen.get("forest_max_tree_density", 0.72))))
    min_tree_density = max(0.0, min(max_tree_density, float(worldgen.get("forest_min_tree_density", 0.16))))

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r][c] in OPEN_WATER_TILE_IDS:
                continue
            density = mask[r][c]
            if density < bush_threshold:
                continue
            biomes[(r, c)] = "forest"
            if density >= tree_threshold:
                scale = (density - tree_threshold) / max(1e-6, 1.0 - tree_threshold)
                tree_prob = min_tree_density + ((max_tree_density - min_tree_density) * scale)
                if rng.random() < tree_prob:
                    grid[r][c] = tree_tile
                elif rng.random() < 0.7:
                    grid[r][c] = bush_tile
                else:
                    grid[r][c] = fallback_floor
            else:
                t = (density - bush_threshold) / max(1e-6, tree_threshold - bush_threshold)
                if rng.random() < (0.75 * t):
                    grid[r][c] = bush_tile
                elif rng.random() < (0.12 * t):
                    grid[r][c] = tree_tile
                else:
                    grid[r][c] = fallback_floor


def _apply_shore_tiles(
    grid: List[List[str]],
    biomes: Dict[Tuple[int, int], str],
    rng: random.Random,
    *,
    dirt_tile: str,
    sand_tile: str,
    worldgen: Dict[str, object],
) -> None:
    if not grid or not grid[0]:
        return
    h = len(grid)
    w = len(grid[0])
    perm = _build_permutation(rng)
    nscale = max(8.0, float(worldgen.get("shore_noise_scale", 20.0)))
    near_water_threshold = max(1, int(worldgen.get("shore_water_neighbor_threshold", 1)))

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            tile_id = grid[r][c]
            if tile_id in OPEN_WATER_TILE_IDS:
                continue
            water_neighbors = 0
            forest_neighbors = 0
            for nr in range(r - 1, r + 2):
                for nc in range(c - 1, c + 2):
                    if nr == r and nc == c:
                        continue
                    nt = grid[nr][nc]
                    if nt in OPEN_WATER_TILE_IDS:
                        water_neighbors += 1
                    if biomes.get((nr, nc), "") == "forest":
                        forest_neighbors += 1
            if water_neighbors < near_water_threshold:
                continue
            biome = biomes.get((r, c), "")
            noise = _fractal_noise_2d(float(c) / nscale, float(r) / nscale, perm, octaves=2)
            sand_bias = 0.15 + (0.08 * float(water_neighbors))
            if biome in {"plains", "rocky"}:
                sand_bias += 0.18
            if forest_neighbors > 0 or biome == "forest":
                sand_bias -= 0.25
            sand_bias += (noise - 0.5) * 0.2
            sand_bias = max(0.05, min(0.92, sand_bias))
            if rng.random() < sand_bias:
                grid[r][c] = sand_tile
                biomes[(r, c)] = "shore_sand"
            else:
                grid[r][c] = dirt_tile
                biomes[(r, c)] = "shore_dirt"


def _apply_dirt_patches(
    grid: List[List[str]],
    biomes: Dict[Tuple[int, int], str],
    rng: random.Random,
    *,
    dirt_tile: str,
    worldgen: Dict[str, object],
) -> None:
    if not grid or not grid[0]:
        return
    h = len(grid)
    w = len(grid[0])
    area = h * w
    patch_scale = max(1, int(worldgen.get("dirt_cluster_scale", 150 * 150)))
    patch_count = max(1, int(area / float(patch_scale)))
    patch_count = max(patch_count, int(worldgen.get("min_dirt_clusters", 2)))
    patch_count = min(patch_count, int(worldgen.get("max_dirt_clusters", 20)))
    patch_radius = max(4, int(worldgen.get("dirt_cluster_radius", 12)))
    perm = _build_permutation(rng)
    nscale = max(6.0, float(worldgen.get("dirt_cluster_noise_scale", 18.0)))
    roughness = max(0.1, min(0.7, float(worldgen.get("dirt_cluster_roughness", 0.28))))
    for _ in range(patch_count):
        cr = rng.randint(2 + patch_radius, max(2 + patch_radius, h - 3 - patch_radius))
        cc = rng.randint(2 + patch_radius, max(2 + patch_radius, w - 3 - patch_radius))
        rr = float(max(1, patch_radius))
        r0 = max(1, cr - patch_radius)
        r1 = min(h - 2, cr + patch_radius)
        c0 = max(1, cc - patch_radius)
        c1 = min(w - 2, cc + patch_radius)
        for r in range(r0, r1 + 1):
            dr = float(r - cr) / rr
            for c in range(c0, c1 + 1):
                if grid[r][c] in OPEN_WATER_TILE_IDS:
                    continue
                if grid[r][c] in {"indestructible_wall", "rock_wall", "wood_wall", "tree"}:
                    continue
                dc = float(c - cc) / rr
                dist = math.sqrt((dr * dr) + (dc * dc))
                if dist > 1.25:
                    continue
                noise = _fractal_noise_2d(float(c) / nscale, float(r) / nscale, perm, octaves=3)
                rf = 1.0 + ((noise - 0.5) * 2.0 * roughness)
                if dist <= rf:
                    grid[r][c] = dirt_tile
                    if biomes.get((r, c), "") not in {"shore_sand", "shore_dirt", "water"}:
                        biomes[(r, c)] = "dirt_patch"


def generate_biome_terrain(
    width: int,
    height: int,
    tiles: Dict[str, TileDef],
    rng: random.Random,
    biome_defs: List[Dict[str, object]],
    wall_tile_id: str,
    floor_fallback_id: str,
    worldgen: Dict[str, object] | None = None,
    structures_defs: List[Dict[str, object]] | None = None,
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
    stone_tile = _valid_tile(tiles, str(worldgen.get("stone_tile_id", "rock_wall")), wall_id)
    dirt_tile = _valid_tile(tiles, str(worldgen.get("dirt_tile_id", "dirt_floor")), floor_id)
    sand_tile = _valid_tile(tiles, str(worldgen.get("sand_tile_id", "sand_floor")), dirt_tile)

    biome_ids: List[str] = []
    biome_weights: List[float] = []
    biome_tile_weight: Dict[str, Tuple[List[str], List[float], List[float]]] = {}
    global_tile_ids = weighted_tile_ids(tiles)
    global_tile_weights = weighted_tile_weights(tiles)
    global_cum = _cum_weights(global_tile_weights if global_tile_weights else [1.0])
    if not global_tile_ids:
        global_tile_ids = [floor_id]
    allow_scattered_open_water = bool(worldgen.get("allow_scattered_open_water", False))
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
                if not allow_scattered_open_water and stid in OPEN_WATER_TILE_IDS:
                    # Keep open-water placement controlled by dedicated lake/river passes.
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
    n_lakes = max(0, int((width * height) / float(lake_scale * lake_scale)))
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
    river_cross_map_probability = float(worldgen.get("river_cross_map_probability", 0.7))
    river_endpoint_margin = int(worldgen.get("river_endpoint_margin", 6))
    river_meander_strength = float(worldgen.get("river_meander_strength", 1.2))
    for _ in range(n_rivers):
        _paint_river(
            grid=grid,
            biomes=biomes,
            rng=rng,
            deep_tile=deep_tile,
            shallow_tile=shallow_tile,
            min_width=river_w_min,
            max_width=river_w_max,
            cross_map_probability=river_cross_map_probability,
            endpoint_margin=river_endpoint_margin,
            meander_strength=river_meander_strength,
        )

    _apply_shore_tiles(
        grid=grid,
        biomes=biomes,
        rng=rng,
        dirt_tile=dirt_tile,
        sand_tile=sand_tile,
        worldgen=worldgen,
    )

    # Dense forest zones and exposed stone/mineral zones.
    area = width * height
    forest_blobs = max(2, int(area / float(max(1, int(worldgen.get("forest_cluster_scale", 130 * 130))))))
    stone_blobs = max(2, int(area / float(max(1, int(worldgen.get("stone_cluster_scale", 180 * 180))))))
    forest_radius = max(8, int(worldgen.get("forest_cluster_radius", 22)))
    stone_radius = max(8, int(worldgen.get("stone_cluster_radius", 18)))

    _apply_forest_density_mask(
        grid=grid,
        biomes=biomes,
        rng=rng,
        forest_blobs=forest_blobs,
        forest_radius=forest_radius,
        tree_tile=tree_tile,
        bush_tile="bush" if "bush" in tiles else tree_tile,
        fallback_floor=floor_id,
        worldgen=worldgen,
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

    _apply_dirt_patches(
        grid=grid,
        biomes=biomes,
        rng=rng,
        dirt_tile=dirt_tile,
        worldgen=worldgen,
    )

    structure_defs = [dict(x) for x in list(structures_defs or []) if isinstance(x, dict)]
    if not structure_defs:
        structure_defs = [dict(x) for x in list(worldgen.get("structures", [])) if isinstance(x, dict)]
    _place_structures(
        grid=grid,
        biomes=biomes,
        tiles=tiles,
        rng=rng,
        structures=structure_defs,
        floor_fallback_id=floor_id,
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
    wall_tile_id: str = "indestructible_wall",
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
