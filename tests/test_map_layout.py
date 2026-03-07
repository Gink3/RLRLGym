import tempfile
import unittest
from pathlib import Path

from rlrlgym.map_layout import load_map_layout


class TestMapLayout(unittest.TestCase):
    def test_load_valid_map_layout(self):
        payload = {
            "schema_version": 1,
            "map": {
                "name": "m",
                "grid": [
                    ["wall", "wall", "wall"],
                    ["wall", "floor", "wall"],
                    ["wall", "wall", "wall"],
                ],
                "biomes": [{"position": [1, 1], "biome": "plains"}],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "map.json"
            p.write_text(__import__("json").dumps(payload), encoding="utf-8")
            layout = load_map_layout(p)
            self.assertEqual(layout.grid[1][1], "floor")
            self.assertEqual(layout.biomes.get((1, 1)), "plains")

    def test_rejects_jagged_grid(self):
        payload = {
            "schema_version": 1,
            "map": {"grid": [["wall", "wall"], ["wall"]]},
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad_map.json"
            p.write_text(__import__("json").dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_map_layout(p)


if __name__ == "__main__":
    unittest.main()
