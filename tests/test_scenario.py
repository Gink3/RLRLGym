import json
import tempfile
import unittest
from pathlib import Path

from rlrlgym import EnvConfig
from rlrlgym.world.env import PettingZooParallelRLRLGym
from rlrlgym.systems.scenario import (
    SCENARIO_AGENTS_FILE,
    SCENARIO_ENV_FILE,
    SUPPORTED_AGENT_POLICIES,
    apply_scenario_to_env_config,
    load_scenario,
    make_all_race_class_combinations,
    save_scenario,
    Scenario,
    ScenarioAgent,
)


class TestScenario(unittest.TestCase):
    def test_supported_policies_include_plain_dqn(self):
        self.assertIn("dqn", SUPPORTED_AGENT_POLICIES)

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
                    {
                        "race": "human",
                        "class": "fighter",
                        "profile": "human",
                        "network": "default",
                        "policy": "ppo_masked",
                        "policy_id": "explorer_shared",
                    },
                    {
                        "race": "orc",
                        "class": "rogue",
                        "profile": "orc",
                        "network": "default",
                        "policy": "ppo_masked",
                        "policy_id": "explorer_shared",
                    },
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
            self.assertEqual(str(cfg.agent_scenario[0].get("policy", "")), "ppo_masked")
            self.assertEqual(
                str(cfg.agent_scenario[0].get("policy_id", "")), "explorer_shared"
            )
            self.assertTrue(str(cfg.agent_scenario[0].get("name", "")).strip())

    def test_rejects_unknown_policy(self):
        payload = {
            "schema_version": 1,
            "scenario": {
                "name": "bad_policy",
                "env_config": {},
                "agents": [
                    {"race": "human", "class": "fighter", "policy": "invalid_policy"}
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "scenario.json"
            p.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_scenario(p)

    def test_save_and_load_policy_field(self):
        policy = sorted(SUPPORTED_AGENT_POLICIES)[0]
        scenario = Scenario(
            name="policy_roundtrip",
            env_config={},
            agents=[
                ScenarioAgent(
                    agent_id="agent_0",
                    race="human",
                    class_name="fighter",
                    policy=policy,
                    policy_id="group_alpha",
                )
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = save_scenario(Path(tmp) / "policy_roundtrip", scenario)
            loaded = load_scenario(out_dir)
            self.assertEqual(loaded.agents[0].policy, policy)
            self.assertEqual(loaded.agents[0].policy_id, "group_alpha")

    def test_default_or_blank_names_are_auto_generated(self):
        scenario = Scenario(
            name="name_gen_case",
            env_config={},
            agents=[
                ScenarioAgent(agent_id="agent_0", race="human", class_name="fighter", name="agent_0"),
                ScenarioAgent(agent_id="agent_1", race="orc", class_name="rogue", name=""),
            ],
        )
        cfg = apply_scenario_to_env_config(EnvConfig(render_enabled=False), scenario)
        name0 = str(cfg.agent_scenario[0].get("name", "")).strip()
        name1 = str(cfg.agent_scenario[1].get("name", "")).strip()
        self.assertTrue(name0 and " " in name0)
        self.assertTrue(name1 and " " in name1)
        self.assertNotEqual(name0, "agent_0")

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

    def test_scenario_can_bundle_spawn_and_structure_tables(self):
        base = Path("data/base")
        bundled_env = {
            "width": 16,
            "height": 12,
            "max_steps": 10,
            # Intentionally bad paths to prove bundled payloads are used.
            "tiles_path": "missing/tiles.json",
            "items_path": "missing/items.json",
            "monsters_path": "missing/monsters.json",
            "monster_spawns_path": "missing/monster_spawns.json",
            "mapgen_config_path": "missing/mapgen_config.json",
            "recipes_path": "missing/recipes.json",
            "statuses_path": "missing/statuses.json",
            "spells_path": "missing/spells.json",
            "enchantments_path": "missing/enchantments.json",
            "structures_data": json.loads((base / "tiles.json").read_text(encoding="utf-8")),
            "items_data": json.loads((base / "items.json").read_text(encoding="utf-8")),
            "monsters_data": json.loads((base / "monsters.json").read_text(encoding="utf-8")),
            "monster_spawns_data": json.loads(
                (base / "monster_spawns.json").read_text(encoding="utf-8")
            ),
            "mapgen_config_data": json.loads(
                (base / "mapgen_config.json").read_text(encoding="utf-8")
            ),
            "recipes_data": json.loads(
                (base / "recipes.json").read_text(encoding="utf-8")
            ),
            "statuses_data": json.loads(
                (base / "statuses.json").read_text(encoding="utf-8")
            ),
            "spells_data": json.loads(
                (base / "spells.json").read_text(encoding="utf-8")
            ),
            "enchantments_data": json.loads(
                (base / "enchantments.json").read_text(encoding="utf-8")
            ),
        }
        scenario = Scenario(
            name="bundled_case",
            env_config=bundled_env,
            agents=[
                ScenarioAgent(agent_id="agent_0", race="human", class_name="fighter"),
                ScenarioAgent(agent_id="agent_1", race="orc", class_name="rogue"),
            ],
        )
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = save_scenario(Path(tmp) / "bundled_case", scenario)
            env = PettingZooParallelRLRLGym(
                EnvConfig(scenario_path=str(out_dir), render_enabled=False)
            )
            obs, _ = env.reset(seed=123)
            self.assertEqual(set(obs.keys()), {"agent_0", "agent_1"})
            self.assertIn("floor", env.tiles)
            self.assertIn("rat", env.monsters)
            self.assertGreater(len(env.monster_spawns), 0)


if __name__ == "__main__":
    unittest.main()
