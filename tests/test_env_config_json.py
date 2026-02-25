import json
import tempfile
import unittest
from pathlib import Path

from rlrlgym.env import EnvConfig


class TestEnvConfigJson(unittest.TestCase):
    def test_load_env_config_from_json(self):
        payload = {
            "schema_version": 1,
            "env_config": {
                "width": 31,
                "height": 17,
                "max_steps": 222,
                "n_agents": 3,
                "render_enabled": False,
                "agent_profile_map": {"agent_0": "human"},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "env_config.json"
            p.write_text(json.dumps(payload), encoding="utf-8")
            cfg = EnvConfig.from_json(p)
        self.assertEqual(cfg.width, 31)
        self.assertEqual(cfg.height, 17)
        self.assertEqual(cfg.max_steps, 222)
        self.assertEqual(cfg.n_agents, 3)
        self.assertFalse(cfg.render_enabled)
        self.assertEqual(cfg.agent_profile_map["agent_0"], "human")


if __name__ == "__main__":
    unittest.main()
