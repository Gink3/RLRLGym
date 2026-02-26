import tempfile
import unittest
from pathlib import Path

from train.trainer import MultiAgentTrainer, TrainConfig


class TestTrainModule(unittest.TestCase):
    def test_training_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = TrainConfig(
                episodes=3,
                max_steps=20,
                seed=4,
                output_dir=str(Path(tmp) / "train_out"),
                width=12,
                height=10,
                n_agents=2,
            )
            trainer = MultiAgentTrainer(cfg)
            result = trainer.train()

            self.assertIn("aggregate", result)
            self.assertIn("artifacts", result)
            self.assertIn("checkpoint", result)
            self.assertIn("replays", result)
            self.assertIn("run_metrics", result["aggregate"])
            self.assertIn("network_parameter_counts", result["aggregate"]["run_metrics"])
            counts = result["aggregate"]["run_metrics"]["network_parameter_counts"]
            self.assertIn("human", counts)
            self.assertIn("orc", counts)
            self.assertGreater(counts["human"], 0)
            self.assertGreater(counts["orc"], 0)
            self.assertTrue(Path(result["checkpoint"]).exists())
            self.assertTrue(result["checkpoint"].endswith("neural_policies.json"))
            self.assertEqual(result["artifacts"], {})
            self.assertEqual(result["replays"], [])

    def test_replay_auto_save_every_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = TrainConfig(
                episodes=2,
                max_steps=5,
                seed=9,
                output_dir=str(Path(tmp) / "train_out"),
                width=10,
                height=8,
                n_agents=2,
                replay_save_every=1,
                show_progress=False,
            )
            trainer = MultiAgentTrainer(cfg)
            result = trainer.train()

            self.assertEqual(len(result["replays"]), 2)
            for replay_path in result["replays"]:
                p = Path(replay_path)
                self.assertTrue(p.exists())
                self.assertTrue(p.name.endswith(".replay.json"))


if __name__ == "__main__":
    unittest.main()
