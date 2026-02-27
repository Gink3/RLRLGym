import json
import tempfile
import unittest
from pathlib import Path

from rlrlgym import EnvConfig
from rlrlgym.env import PettingZooParallelRLRLGym
from rlrlgym.scenario import (
    SCENARIO_AGENTS_FILE,
    SCENARIO_ENV_FILE,
    apply_scenario_to_env_config,
    load_scenario,
    make_all_race_class_combinations,
    save_scenario,
    Scenario,
    ScenarioAgent,
)


class TestScenario(unittest.TestCase):
    def test_make_all_race_class_combinations(self):
        agents = make_all_race_class_combinations(["human", "orc"], ["fighter", "rogue"])
        self.assertEqual(len(agents), 4)
        self.assertEqual(agents[0].agent_id, "agent_0")
        self.assertEqual(agents[-1].agent_id, "agent_3")

    def test_load_and_apply_scenario(self):
        payload = {
            "schema_version": 1,
            "scenario": {
                "name": "smoke",
                "env_config": {"width": 22, "height": 18},
                "agents": [
                    {"race": "human", "class": "fighter", "profile": "human", "network": "default"},
                    {"race": "orc", "class": "rogue", "profile": "orc", "network": "default"},
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "scenario.json"
            p.write_text(json.dumps(payload), encoding="utf-8")
            scenario = load_scenario(p)
            cfg = apply_scenario_to_env_config(EnvConfig(render_enabled=False), scenario)

            self.assertEqual(cfg.n_agents, 2)
            self.assertEqual(cfg.width, 22)
            self.assertEqual(cfg.height, 18)
            self.assertEqual(cfg.agent_race_map["agent_0"], "human")
            self.assertEqual(cfg.agent_class_map["agent_1"], "rogue")
            self.assertEqual(cfg.agent_profile_map["agent_1"], "orc")

    def test_env_config_scenario_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "scenario.json"
            p.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "scenario": {
                            "name": "env_boot",
                            "env_config": {"width": 16, "height": 12, "max_steps": 10},
                            "agents": [
                                {"race": "human", "class": "fighter", "profile": "human"},
                                {"race": "orc", "class": "rogue", "profile": "orc"},
                            ],
                        },
                    }
                ),
                encoding="utf-8",
            )
            env = PettingZooParallelRLRLGym(EnvConfig(scenario_path=str(p), render_enabled=False))
            self.assertEqual(env.config.n_agents, 2)
            self.assertEqual(env.config.width, 16)
            self.assertEqual(env.config.height, 12)
            self.assertEqual(len(env.possible_agents), 2)

    def test_save_and_load_split_scenario_dir(self):
        scenario = Scenario(
            name="split_case",
            env_config={"width": 20, "height": 14, "max_steps": 50},
            agents=[
                ScenarioAgent(
                    agent_id="agent_0",
                    race="human",
                    class_name="fighter",
                    name="Atlas",
                    profile="reward_explorer_policy_v1",
                ),
                ScenarioAgent(agent_id="agent_1", race="orc", class_name="rogue", profile="orc"),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = save_scenario(Path(tmp) / "split_case", scenario)
            self.assertTrue((out_dir / SCENARIO_ENV_FILE).exists())
            self.assertTrue((out_dir / SCENARIO_AGENTS_FILE).exists())
            loaded = load_scenario(out_dir)
            self.assertEqual(loaded.name, "split_case")
            self.assertEqual(len(loaded.agents), 2)
            self.assertEqual(loaded.agents[0].name, "Atlas")
            self.assertEqual(loaded.env_config.get("width"), 20)


if __name__ == "__main__":
    unittest.main()
