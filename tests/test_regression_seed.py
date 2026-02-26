import unittest

from rlrlgym import EnvConfig, PettingZooParallelRLRLGym
from rlrlgym.constants import ACTION_WAIT


class TestRegressionSeed(unittest.TestCase):
    def test_fixed_seed_regression_signature(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=10, height=8, n_agents=2, max_steps=10)
        )
        env.reset(seed=123)

        reward_trace = []
        for _ in range(4):
            _, rewards, _, _, _ = env.step(
                {"agent_0": ACTION_WAIT, "agent_1": ACTION_WAIT}
            )
            reward_trace.append(
                (round(rewards["agent_0"], 3), round(rewards["agent_1"], 3))
            )

        signature = (
            tuple(env.state.grid[1][:5]),
            tuple(env.state.grid[2][:5]),
            tuple(reward_trace),
        )

        expected = (
            ("wall", "floor", "floor", "floor", "floor"),
            ("wall", "water", "floor", "floor", "floor"),
            ((-0.005, -0.008), (-0.005, -0.008), (-0.125, -0.008), (-0.12, -0.036)),
        )
        self.assertEqual(signature, expected)


if __name__ == "__main__":
    unittest.main()
