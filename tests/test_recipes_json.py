import json
import tempfile
import unittest
from pathlib import Path

from rlrlgym.content.recipes import load_recipes


class TestRecipesJson(unittest.TestCase):
    def test_load_recipes(self):
        payload = {
            "schema_version": 1,
            "recipes": [
                {
                    "id": "smelt_test",
                    "inputs": {"copper_ore": 2},
                    "outputs": {"copper_ingot": 1},
                    "skill": "smithing",
                    "min_skill": 0,
                    "station": "smelter",
                    "required_tool_category": "pickaxe",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "recipes.json"
            p.write_text(json.dumps(payload), encoding="utf-8")
            out = load_recipes(p)
            self.assertIn("smelt_test", out)
            self.assertEqual(out["smelt_test"].inputs["copper_ore"], 2)
            self.assertEqual(out["smelt_test"].required_tool_category, "pickaxe")

    def test_requires_outputs_or_build_tile(self):
        payload = {
            "schema_version": 1,
            "recipes": [{"id": "bad", "inputs": {"wood": 2}, "outputs": {}}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "recipes.json"
            p.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_recipes(p)

    def test_allows_build_station_recipe(self):
        payload = {
            "schema_version": 1,
            "recipes": [
                {
                    "id": "build_workbench_test",
                    "inputs": {"wood": 2},
                    "outputs": {},
                    "build_station_id": "workbench",
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "recipes.json"
            p.write_text(json.dumps(payload), encoding="utf-8")
            out = load_recipes(p)
            self.assertEqual(out["build_workbench_test"].build_station_id, "workbench")


if __name__ == "__main__":
    unittest.main()
