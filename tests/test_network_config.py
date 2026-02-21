import json
import tempfile
import unittest
from pathlib import Path

from train.network_config import load_network_configs


class TestNetworkConfig(unittest.TestCase):
    def test_load_network_configs(self):
        data = {
            "schema_version": 1,
            "architectures": [
                {
                    "name": "human",
                    "hidden_layers": [32, 16],
                    "activation": "relu",
                    "learning_rate": 0.003,
                    "gamma": 0.97,
                    "epsilon_start": 0.2,
                    "epsilon_end": 0.05,
                    "epsilon_decay": 0.99,
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "nets.json"
            p.write_text(json.dumps(data), encoding="utf-8")
            out = load_network_configs(p)
            self.assertIn("human", out)
            self.assertEqual(out["human"].hidden_layers, [32, 16])

    def test_requires_schema_version(self):
        data = {"architectures": []}
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "nets.json"
            p.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_network_configs(p)


if __name__ == "__main__":
    unittest.main()
