import json
import tempfile
import unittest
from pathlib import Path

from rlrlgym.items import load_items


class TestItemsJson(unittest.TestCase):
    def test_requires_schema_version(self):
        bad = {"items": [], "chest_loot_table": []}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "items.json"
            p.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_items(p)

    def test_loads_valid_schema(self):
        good = {
            "schema_version": 1,
            "items": [
                {
                    "id": "dagger",
                    "weight": 1.0,
                    "weapon": {
                        "damage_type": "pierce",
                        "damage_min": 2,
                        "damage_max": 4,
                        "skill": "melee",
                    },
                },
                {
                    "id": "leather_cap",
                    "weight": 0.6,
                    "armor_slot": "head",
                    "dr_bonus_vs": {"slash": 0, "pierce": 0, "blunt": 0},
                },
                {"id": "coin", "weight": 0.02, "is_treasure": True},
            ],
            "chest_loot_table": ["dagger", "leather_cap", "coin"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "items.json"
            p.write_text(json.dumps(good), encoding="utf-8")
            items = load_items(p)
            self.assertIn("dagger", items.weapon_damage_type)
            self.assertEqual(items.weapon_damage_type["dagger"], "pierce")
            self.assertIn("leather_cap", items.armor_slot_by_item)
            self.assertEqual(items.armor_slot_by_item["leather_cap"], "head")
            self.assertIn("coin", items.treasure_items)

    def test_chest_loot_references_known_items(self):
        bad = {
            "schema_version": 1,
            "items": [{"id": "ration", "weight": 1.0, "edible_hunger": 8}],
            "chest_loot_table": ["missing_item"],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "items.json"
            p.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_items(p)


if __name__ == "__main__":
    unittest.main()
