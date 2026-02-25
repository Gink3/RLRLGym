import json
import tempfile
import unittest
from pathlib import Path

from rlrlgym.races import load_races


class TestRacesJson(unittest.TestCase):
    def test_requires_schema_version(self):
        bad = {"races": []}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "agent_races.json"
            p.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_races(p)

    def test_loads_valid_schema(self):
        good = {
            "schema_version": 1,
            "races": [
                {
                    "name": "human",
                    "strength": 5,
                    "dexterity": 6,
                    "intellect": 5,
                    "base_dr_min": 0,
                    "base_dr_max": 1,
                    "dr_bonus_vs": {"slash": 0, "pierce": 0, "blunt": 0},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "agent_races.json"
            p.write_text(json.dumps(good), encoding="utf-8")
            races = load_races(p)
            self.assertIn("human", races)
            self.assertEqual(races["human"].strength, 5)
            self.assertEqual(races["human"].dr_bonus_vs["pierce"], 0)


if __name__ == "__main__":
    unittest.main()
