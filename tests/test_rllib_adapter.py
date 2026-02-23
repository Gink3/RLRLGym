import unittest

from rlrlgym.featurize import observation_vector_size
from rlrlgym.rllib_env import RLRLGymRLlibEnv

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None
try:
    import gymnasium  # noqa: F401
except Exception:  # pragma: no cover
    gymnasium = None


@unittest.skipIf(np is None or gymnasium is None, "numpy/gymnasium not installed")
class TestRLlibAdapter(unittest.TestCase):
    def test_reset_and_step_shapes(self):
        env = RLRLGymRLlibEnv({"width": 12, "height": 10, "n_agents": 2, "max_steps": 5})
        obs, info = env.reset(seed=3)
        self.assertIn("agent_0", obs)
        self.assertIsInstance(obs["agent_0"], np.ndarray)
        self.assertEqual(obs["agent_0"].shape, (observation_vector_size(),))

        actions = {aid: 4 for aid in obs.keys()}
        next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
        self.assertIn("agent_0", rewards)
        self.assertIn("__all__", terminateds)
        self.assertIn("__all__", truncateds)
        self.assertIsInstance(next_obs, dict)
        self.assertIsInstance(infos, dict)


if __name__ == "__main__":
    unittest.main()
