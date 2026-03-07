import random
import unittest

from rlrlgym.world.mapgen import generate_biome_terrain
from rlrlgym.world.mapgen_config import load_mapgen_config
from rlrlgym.content.tiles import load_tileset


class TestMapgenShapeControls(unittest.TestCase):
    def test_inland_river_can_avoid_map_edges(self):
        tiles = load_tileset("data/base/tiles.json")
        cfg = load_mapgen_config("data/base/mapgen_config.json")
        worldgen = dict(cfg.worldgen)
        worldgen.update(
            {
                "min_lakes": 0,
                "max_lakes": 0,
                "min_rivers": 1,
                "max_rivers": 1,
                "river_min_width": 2,
                "river_max_width": 2,
                "river_cross_map_probability": 0.0,
                "river_endpoint_margin": 10,
            }
        )
        width = 80
        height = 60
        grid, _ = generate_biome_terrain(
            width=width,
            height=height,
            tiles=tiles,
            rng=random.Random(123),
            biome_defs=cfg.biomes,
            wall_tile_id=cfg.wall_tile_id,
            floor_fallback_id=cfg.floor_fallback_id,
            worldgen=worldgen,
            structures_defs=[],
            min_width=cfg.min_width,
            min_height=cfg.min_height,
        )
        water = [
            (r, c)
            for r in range(height)
            for c in range(width)
            if grid[r][c] in {"water", "shallow_water", "deep_water"}
        ]
        self.assertGreater(len(water), 0)
        safe_band = int(worldgen["river_endpoint_margin"]) - int(worldgen["river_max_width"]) - 1
        near_edge = [
            (r, c)
            for (r, c) in water
            if (
                r <= safe_band
                or c <= safe_band
                or r >= (height - 1 - safe_band)
                or c >= (width - 1 - safe_band)
            )
        ]
        self.assertEqual(len(near_edge), 0)


if __name__ == "__main__":
    unittest.main()
