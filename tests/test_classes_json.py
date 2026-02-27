import json
import tempfile
import unittest
from pathlib import Path

from rlrlgym.classes import load_classes


class TestClassesJson(unittest.TestCase):
    def test_requires_schema_version(self):
        bad = {"classes": []}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "agent_classes.json"
            p.write_text(json.dumps(bad), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_classes(p)

    def test_loads_valid_schema(self):
        good = {
            "schema_version": 1,
            "classes": [
                {
                    "name": "fighter",
                    "starting_items": ["ration"],
                    "skill_modifiers": {"melee": 2},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "agent_classes.json"
            p.write_text(json.dumps(good), encoding="utf-8")
            classes = load_classes(p)
            self.assertIn("fighter", classes)
            self.assertIn("ration", classes["fighter"].starting_items)
            self.assertEqual(classes["fighter"].skill_modifiers["melee"], 2)


if __name__ == "__main__":
    unittest.main()
