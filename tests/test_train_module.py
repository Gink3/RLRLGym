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
            self.assertTrue(Path(result["checkpoint"]).exists())
            self.assertTrue(result["checkpoint"].endswith("neural_policies.json"))
            self.assertTrue(Path(result["artifacts"]["summary"]).exists())


if __name__ == "__main__":
    unittest.main()
