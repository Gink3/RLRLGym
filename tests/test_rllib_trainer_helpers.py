import unittest

from train.rllib_trainer import RLlibTrainer


class TestRLlibTrainerHelpers(unittest.TestCase):
    def test_prefer_nonzero_metric_keeps_last_nonzero_value(self):
        self.assertEqual(RLlibTrainer._prefer_nonzero_metric(0.0, 2.5), 2.5)
        self.assertEqual(RLlibTrainer._prefer_nonzero_metric(1.25, 2.5), 1.25)
        self.assertEqual(RLlibTrainer._prefer_nonzero_metric(float("nan"), 3.0), 3.0)


if __name__ == "__main__":
    unittest.main()
