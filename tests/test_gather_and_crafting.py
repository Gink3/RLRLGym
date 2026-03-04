import unittest

from rlrlgym import EnvConfig, PettingZooParallelRLRLGym
from rlrlgym.constants import ACTION_INTERACT
from rlrlgym.models import ResourceNodeState, StationState


class TestGatherAndCrafting(unittest.TestCase):
    def test_interact_gathers_resource_node(self):
        cfg = EnvConfig(width=12, height=10, n_agents=1, max_steps=5, render_enabled=False)
        env = PettingZooParallelRLRLGym(cfg)
        env.reset(seed=7)
        aid = "agent_0"
        pos = env.state.agents[aid].position
        env.state.resource_nodes[pos] = ResourceNodeState(
            node_id="stone_vein",
            position=pos,
            skill="mining",
            drop_item="stone",
            remaining=3,
            max_yield=3,
            biome="rocky",
        )
        _, _, _, _, info = env.step({aid: ACTION_INTERACT})
        self.assertTrue(
            any(str(evt).startswith("gather:stone_vein:stone:") for evt in info[aid]["events"])
        )
        self.assertIn("stone", env.state.agents[aid].inventory)

    def test_interact_crafts_at_station(self):
        recipes_data = {
            "schema_version": 1,
            "recipes": [
                {
                    "id": "mint_test",
                    "inputs": {"copper_ingot": 1},
                    "outputs": {"copper_coin": 4},
                    "skill": "smithing",
                    "station": "mint",
                }
            ],
        }
        cfg = EnvConfig(
            width=12,
            height=10,
            n_agents=1,
            max_steps=5,
            render_enabled=False,
            recipes_data=recipes_data,
        )
        env = PettingZooParallelRLRLGym(cfg)
        env.reset(seed=9)
        aid = "agent_0"
        pos = env.state.agents[aid].position
        env.state.stations[pos] = StationState(
            station_id="mint",
            position=pos,
            speed_multiplier=1.0,
            quality_tier=0,
            unlock_recipes=["mint_test"],
        )
        env.state.agents[aid].inventory.append("copper_ingot")
        _, _, _, _, info = env.step({aid: ACTION_INTERACT})
        self.assertTrue(
            any(str(evt).startswith("craft:mint_test:") for evt in info[aid]["events"])
        )
        self.assertGreaterEqual(env.state.agents[aid].inventory.count("copper_coin"), 4)


if __name__ == "__main__":
    unittest.main()
