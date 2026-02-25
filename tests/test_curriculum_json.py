import json
import tempfile
import unittest
from pathlib import Path

from rlrlgym.curriculum import load_curriculum_phases


class TestCurriculumJson(unittest.TestCase):
    def test_requires_schema_version(self):
        bad = {"phases": []}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "curriculum_phases.json"
            p.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_curriculum_phases(p)

    def test_loads_valid_schema(self):
        good = {
            "schema_version": 1,
            "phases": [
                {
                    "name": "p1",
                    "until_episode": 100,
                    "width": 10,
                    "height": 10,
                    "max_steps": 80,
                    "monster_density": 0.0,
                    "chest_density": 0.08,
                },
                {
                    "name": "p2",
                    "until_episode": 0,
                    "width": 14,
                    "height": 14,
                    "max_steps": 120,
                    "monster_density": 0.03,
                    "chest_density": 0.05,
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "curriculum_phases.json"
            p.write_text(json.dumps(good), encoding="utf-8")
            phases = load_curriculum_phases(p)
            self.assertEqual(len(phases), 2)
            self.assertEqual(phases[0]["name"], "p1")
            self.assertEqual(phases[1]["monster_density"], 0.03)


if __name__ == "__main__":
    unittest.main()
