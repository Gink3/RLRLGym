import json
import tempfile
import unittest
from pathlib import Path

from rlrlgym.tiles import load_tileset


class TestTiles(unittest.TestCase):
    def test_requires_schema_version(self):
        bad = {"tiles": []}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tiles.json"
            p.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_tileset(p)

    def test_loads_valid_schema(self):
        good = {
            "schema_version": 1,
            "tiles": [
                {
                    "id": "floor",
                    "glyph": ".",
                    "color": "white",
                    "walkable": True,
                    "spawn_weight": 1.0,
                    "max_interactions": 0,
                    "loot_table": [],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "tiles.json"
            p.write_text(json.dumps(good), encoding="utf-8")
            tiles = load_tileset(p)
            self.assertIn("floor", tiles)


if __name__ == "__main__":
    unittest.main()
