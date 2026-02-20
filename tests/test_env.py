import unittest

from rlrlgym import EnvConfig, MultiAgentRLRLGym
from rlrlgym.constants import ACTION_MOVE_NORTH, ACTION_WAIT


class TestEnv(unittest.TestCase):
    def test_reset_and_basic_step_shapes(self):
        env = MultiAgentRLRLGym(EnvConfig(width=12, height=10, n_agents=2, max_steps=5))
        obs, info = env.reset(seed=1)
        self.assertEqual(set(obs.keys()), {"agent_0", "agent_1"})
        self.assertTrue(info["agent_0"]["alive"])

        next_obs, rewards, terms, truncs, step_info = env.step({"agent_0": ACTION_WAIT, "agent_1": ACTION_WAIT})
        self.assertIn("agent_0", rewards)
        self.assertIn("agent_1", rewards)
        self.assertFalse(terms["agent_0"])
        self.assertFalse(truncs["agent_0"])
        self.assertIn("events", step_info["agent_0"])
        self.assertIn("agent_0", next_obs)

    def test_wall_blocks_movement(self):
        env = MultiAgentRLRLGym(EnvConfig(width=10, height=8, n_agents=1, max_steps=5))
        env.reset(seed=2)
        pos = env.state.agents["agent_0"].position

        collision_seen = False
        for _ in range(10):
            _, rewards, _, _, info = env.step({"agent_0": ACTION_MOVE_NORTH})
            if "bump" in info["agent_0"]["events"]:
                collision_seen = True
                self.assertLess(rewards["agent_0"], 0)
                break

        self.assertTrue(collision_seen)
        self.assertEqual(env.state.agents["agent_0"].position[1], pos[1])

    def test_rendering_can_be_disabled(self):
        env = MultiAgentRLRLGym(EnvConfig(width=10, height=8, n_agents=1, render_enabled=False))
        env.reset(seed=3)
        self.assertEqual(env.render(color=False), "")


if __name__ == "__main__":
    unittest.main()
