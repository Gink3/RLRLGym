import random
import unittest

from rlrlgym.world.mapgen import generate_biome_terrain
from rlrlgym.world.mapgen_config import load_mapgen_config
from rlrlgym.content.structures import load_structures_config
from rlrlgym.content.tiles import load_tileset


class TestMapgenStructures(unittest.TestCase):
    def test_ruins_structures_are_painted(self):
        tiles = load_tileset("data/base/tiles.json")
        cfg = load_mapgen_config("data/base/mapgen_config.json")
        structures = load_structures_config("data/base/structures.json")
        grid, biomes = generate_biome_terrain(
            width=64,
            height=64,
            tiles=tiles,
            rng=random.Random(12345),
            biome_defs=cfg.biomes,
            wall_tile_id=cfg.wall_tile_id,
            floor_fallback_id=cfg.floor_fallback_id,
            worldgen=cfg.worldgen,
            structures_defs=structures,
            min_width=cfg.min_width,
            min_height=cfg.min_height,
        )
        self.assertEqual(len(grid), 64)
        self.assertEqual(len(grid[0]), 64)
        ruins = [(r, c) for (r, c), b in biomes.items() if b == "ruins"]
        self.assertGreater(len(ruins), 0)


if __name__ == "__main__":
    unittest.main()
