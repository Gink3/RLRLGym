import tempfile
import unittest
from pathlib import Path

from rlrlgym import EnvConfig, PettingZooParallelRLRLGym, TrainingLogger
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
                agent_profile_map={"agent_0": "human", "agent_1": "orc"},
            )
        )
        obs, info = env.reset(seed=1)

        self.assertEqual(obs["agent_0"]["profile"], "human")
        self.assertEqual(obs["agent_1"]["profile"], "orc")
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

    def test_training_logger_outputs(self):
        logger = TrainingLogger(output_dir="outputs_test")
        logger.start_episode(["agent_0", "agent_1"])
        logger.log_step(
            rewards={"agent_0": 0.2, "agent_1": -0.1},
            terminations={"agent_0": False, "agent_1": False},
            truncations={"agent_0": False, "agent_1": False},
            info={"agent_0": {"events": []}, "agent_1": {"events": []}},
            actions={"agent_0": 4, "agent_1": 7},
        )
        logger.log_step(
            rewards={"agent_0": 0.1, "agent_1": -1.0},
            terminations={"agent_0": False, "agent_1": True},
            truncations={"agent_0": False, "agent_1": False},
            info={"agent_0": {"events": []}, "agent_1": {"events": ["death", "starve_tick"]}},
            actions={"agent_0": 5, "agent_1": 4},
        )
        logger.end_episode(step_count=2, alive_agents={"agent_0": True, "agent_1": False})

        agg = logger.aggregate_metrics()
        self.assertEqual(agg["episodes"], 1)
        self.assertGreater(agg["mean_team_return"], -1.0)
        self.assertIn("starvation", agg["cause_of_death_histogram"])
        self.assertEqual(agg["action_histogram"]["wait"], 2)
        self.assertEqual(agg["action_histogram"]["pickup"], 1)
        self.assertEqual(agg["action_histogram"]["loot"], 1)

        with tempfile.TemporaryDirectory() as tmp:
            logger.output_dir = str(Path(tmp) / "out")
            paths = logger.write_outputs()
            self.assertTrue(Path(paths["csv"]).exists())
            self.assertTrue(Path(paths["jsonl"]).exists())
            self.assertTrue(Path(paths["summary"]).exists())
            self.assertTrue(Path(paths["html"]).exists())
            csv_text = Path(paths["csv"]).read_text(encoding="utf-8")
            self.assertIn("action_wait", csv_text)
            self.assertIn("action_pickup", csv_text)
            jsonl_text = Path(paths["jsonl"]).read_text(encoding="utf-8")
            self.assertIn("\"action_counts\"", jsonl_text)


if __name__ == "__main__":
    unittest.main()
