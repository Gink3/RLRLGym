import unittest

from train.rllib_trainer import RLlibTrainer


class TestRLlibTrainerHelpers(unittest.TestCase):
    def test_prefer_nonzero_metric_keeps_last_nonzero_value(self):
        self.assertEqual(RLlibTrainer._prefer_nonzero_metric(0.0, 2.5), 2.5)
        self.assertEqual(RLlibTrainer._prefer_nonzero_metric(1.25, 2.5), 1.25)
        self.assertEqual(RLlibTrainer._prefer_nonzero_metric(float("nan"), 3.0), 3.0)

    def test_extract_custom_metric_map_ignores_nan_values(self):
        trainer = RLlibTrainer.__new__(RLlibTrainer)
        result = {
            "custom_metrics": {
                "death_by_monster__wolf_mean": float("nan"),
                "death_by_monster__boar_mean": 0.5,
            }
        }

        metrics = trainer._extract_custom_metric_map(result)

        self.assertNotIn("death_by_monster__wolf", metrics)
        self.assertEqual(metrics["death_by_monster__boar"], 0.5)


if __name__ == "__main__":
    unittest.main()
