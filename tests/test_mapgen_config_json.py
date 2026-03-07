import json
import tempfile
import unittest
from pathlib import Path

from rlrlgym.world.mapgen_config import load_mapgen_config


class TestMapgenConfigJson(unittest.TestCase):
    def test_requires_schema_version(self):
        bad = {"mapgen": {}}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "mapgen_config.json"
            p.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_mapgen_config(p)

    def test_loads_valid_schema(self):
        good = {
            "schema_version": 1,
            "mapgen": {
                "wall_tile_id": "wall",
                "floor_fallback_id": "floor",
                "chest_density": 0.05,
                "animal_density": 0.02,
                "min_width": 6,
                "min_height": 5,
                "structures": [
                    {
                        "id": "ruins",
                        "density": 0.01,
                        "min_count": 1,
                        "max_count": 2,
                    }
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "mapgen_config.json"
            p.write_text(json.dumps(good), encoding="utf-8")
            cfg = load_mapgen_config(p)
            self.assertEqual(cfg.wall_tile_id, "wall")
            self.assertEqual(cfg.floor_fallback_id, "floor")
            self.assertEqual(cfg.chest_density, 0.05)
            self.assertEqual(cfg.animal_density, 0.02)
            self.assertEqual(cfg.min_width, 6)
            self.assertEqual(cfg.min_height, 5)
            self.assertEqual(len(cfg.structures), 1)


if __name__ == "__main__":
    unittest.main()
