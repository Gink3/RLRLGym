import json
import tempfile
import unittest
from pathlib import Path

from rlrlgym.content.profiles import AgentProfile, load_profiles


class TestProfilesJson(unittest.TestCase):
    def test_load_profiles_success(self):
        data = {
            "schema_version": 1,
            "profiles": [
                {
                    "name": "human",
                    "max_hp": 10,
                    "max_hunger": 20,
                    "view_width": 12,
                    "view_height": 12,
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

    def test_reward_adjustment_supports_crafting_signals(self):
        profile = AgentProfile(
            name="artisan_test",
            max_hp=10,
            max_hunger=30,
            view_width=8,
            view_height=8,
            reward_weights={
                "craft": 0.1,
                "craft_quality": 0.2,
                "skill_up": 0.05,
            },
        )
        delta = profile.reward_adjustment(
            events=[
                "craft:smelt_copper_ingot:station=smelter:speed=1.00",
                "craft_quality:tier=1:bonus_units=1",
                "skill_up:smithing:1",
            ],
            died=False,
        )
        self.assertAlmostEqual(delta, 0.35, places=6)


if __name__ == "__main__":
    unittest.main()
