import json
import tempfile
import unittest
from pathlib import Path

from rlrlgym.profiles import load_profiles


class TestProfilesJson(unittest.TestCase):
    def test_load_profiles_success(self):
        data = {
            "schema_version": 1,
            "profiles": [
                {
                    "name": "human",
                    "max_hp": 10,
                    "max_hunger": 20,
                    "view_radius": 3,
                    "include_grid": True,
                    "include_stats": True,
                    "include_inventory": True,
                    "reward_weights": {"explore": 0.01},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "profiles.json"
            p.write_text(json.dumps(data), encoding="utf-8")
            profiles = load_profiles(p)
            self.assertIn("human", profiles)

    def test_requires_schema_version(self):
        data = {"profiles": []}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "profiles.json"
            p.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_profiles(p)


if __name__ == "__main__":
    unittest.main()
