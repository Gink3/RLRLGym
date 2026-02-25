import json
import tempfile
import unittest
from pathlib import Path

from rlrlgym.monsters import load_monster_spawns, load_monsters


class TestMonstersJson(unittest.TestCase):
    def test_load_monsters_success(self):
        data = {
            "schema_version": 1,
            "monsters": [
                {
                    "id": "rat",
                    "name": "Rat",
                    "symbol": "r",
                    "color": "bright_black",
                    "threat": 1,
                    "hp": 3,
                    "acc": 0,
                    "eva": 2,
                    "dmg_min": 1,
                    "dmg_max": 2,
                    "dr_min": 0,
                    "dr_max": 0,
                    "loot": [{"item": "fruit", "weight": 10, "min_qty": 1, "max_qty": 1}],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "monsters.json"
            p.write_text(json.dumps(data), encoding="utf-8")
            monsters = load_monsters(p)
            self.assertIn("rat", monsters)
            self.assertEqual(monsters["rat"].symbol, "r")

    def test_requires_unique_symbol_color_combo(self):
        data = {
            "schema_version": 1,
            "monsters": [
                {
                    "id": "m1",
                    "name": "M1",
                    "symbol": "x",
                    "color": "red",
                    "threat": 1,
                    "hp": 3,
                    "acc": 0,
                    "eva": 0,
                    "dmg_min": 1,
                    "dmg_max": 1,
                    "dr_min": 0,
                    "dr_max": 0,
                    "loot": [],
                },
                {
                    "id": "m2",
                    "name": "M2",
                    "symbol": "x",
                    "color": "red",
                    "threat": 1,
                    "hp": 3,
                    "acc": 0,
                    "eva": 0,
                    "dmg_min": 1,
                    "dmg_max": 1,
                    "dr_min": 0,
                    "dr_max": 0,
                    "loot": [],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "monsters.json"
            p.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_monsters(p)

    def test_spawn_table_references_known_monsters(self):
        monsters_data = {
            "schema_version": 1,
            "monsters": [
                {
                    "id": "rat",
                    "name": "Rat",
                    "symbol": "r",
                    "color": "bright_black",
                    "threat": 1,
                    "hp": 3,
                    "acc": 0,
                    "eva": 2,
                    "dmg_min": 1,
                    "dmg_max": 2,
                    "dr_min": 0,
                    "dr_max": 0,
                    "loot": [],
                }
            ],
        }
        spawns_data = {
            "schema_version": 1,
            "spawns": [{"monster_id": "rat", "weight": 10}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            mp = Path(tmp) / "monsters.json"
            sp = Path(tmp) / "monster_spawns.json"
            mp.write_text(json.dumps(monsters_data), encoding="utf-8")
            sp.write_text(json.dumps(spawns_data), encoding="utf-8")
            monsters = load_monsters(mp)
            spawns = load_monster_spawns(sp, monsters)
            self.assertEqual(len(spawns), 1)
            self.assertEqual(spawns[0].monster_id, "rat")


if __name__ == "__main__":
    unittest.main()
