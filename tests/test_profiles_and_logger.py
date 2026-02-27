import unittest

from rlrlgym import EnvConfig, PettingZooParallelRLRLGym
from rlrlgym.constants import ACTION_WAIT


class TestProfilesAndLogger(unittest.TestCase):
    def test_human_orc_profile_observation_and_reward_difference(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(
                width=12,
                height=10,
                n_agents=2,
                max_steps=5,
                render_enabled=False,
                agent_profile_map={
                    "agent_0": "reward_explorer_policy_v1",
                    "agent_1": "reward_brawler_policy_v1",
                },
            )
        )
        obs, info = env.reset(seed=1)

        self.assertEqual(obs["agent_0"]["profile"], "reward_explorer_policy_v1")
        self.assertEqual(obs["agent_1"]["profile"], "reward_brawler_policy_v1")
        self.assertEqual(len(obs["agent_0"]["local_tiles"]), 12)
        self.assertEqual(len(obs["agent_0"]["local_tiles"][0]), 12)
        self.assertEqual(len(obs["agent_1"]["local_tiles"]), 10)
        self.assertEqual(len(obs["agent_1"]["local_tiles"][0]), 10)
        self.assertIn("nearby_item_counts", obs["agent_0"]["stats"])
        self.assertIn("tile_interaction_counts", obs["agent_0"]["stats"])
        self.assertIn("teammate_distance", obs["agent_0"]["stats"])

        _, rewards, _, _, info = env.step({"agent_0": ACTION_WAIT, "agent_1": ACTION_WAIT})
        self.assertLess(rewards["agent_1"], rewards["agent_0"])
        self.assertIn("wait", info["agent_0"]["events"])

if __name__ == "__main__":
    unittest.main()
