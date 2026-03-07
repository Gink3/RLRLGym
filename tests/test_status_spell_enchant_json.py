import json
import tempfile
import unittest
from pathlib import Path

from rlrlgym.content.enchantments import load_enchantments
from rlrlgym.content.spells import load_spells
from rlrlgym.content.statuses import load_statuses


class TestStatusSpellEnchantJson(unittest.TestCase):
    def test_load_statuses(self):
        payload = {"schema_version": 1, "statuses": [{"id": "poison", "duration": 3}]}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "statuses.json"
            p.write_text(json.dumps(payload), encoding="utf-8")
            out = load_statuses(p)
            self.assertIn("poison", out)

    def test_load_spells(self):
        payload = {
            "schema_version": 1,
            "spells": [{"id": "zap", "mana_cost": 1, "effects": [{"type": "damage", "amount": 1}]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "spells.json"
            p.write_text(json.dumps(payload), encoding="utf-8")
            out = load_spells(p)
            self.assertIn("zap", out)

    def test_load_enchantments(self):
        payload = {
            "schema_version": 1,
            "enchantments": [{"id": "damage_plus", "effects": [{"type": "damage_plus", "amount": 1}]}],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "ench.json"
            p.write_text(json.dumps(payload), encoding="utf-8")
            out = load_enchantments(p)
            self.assertIn("damage_plus", out)


if __name__ == "__main__":
    unittest.main()
