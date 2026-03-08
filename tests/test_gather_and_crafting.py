import unittest

from rlrlgym import EnvConfig, PettingZooParallelRLRLGym
from rlrlgym.systems.constants import ACTION_EQUIP, ACTION_INTERACT, ACTION_INTERACT_MINE, ACTION_INTERACT_STATION, ACTION_MOVE_EAST
from rlrlgym.systems.models import ResourceNodeState, StationState


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

    def test_mine_tree_harvests_sticks_and_can_fell_with_axe(self):
        cfg = EnvConfig(width=12, height=10, n_agents=1, max_steps=10, render_enabled=False)
        env = PettingZooParallelRLRLGym(cfg)
        env.reset(seed=14)
        aid = "agent_0"
        env.state.agents[aid].position = (5, 5)
        env.state.grid[5][5] = "grass"
        env.state.grid[5][6] = "tree"
        env.state.agents[aid].equipped.append("axe")

        for _ in range(4):
            env.step({aid: ACTION_INTERACT_MINE})
            if env.state.grid[5][6] != "tree":
                break

        inv = env.state.agents[aid].inventory
        self.assertIn("stick", inv)
        self.assertIn("log", inv)
        self.assertNotEqual(env.state.grid[5][6], "tree")

    def test_station_recipe_with_required_tool_category(self):
        recipes_data = {
            "schema_version": 1,
            "recipes": [
                {
                    "id": "plank_test",
                    "inputs": {"log": 1},
                    "outputs": {"wood_plank": 2},
                    "skill": "crafting",
                    "station": "workbench",
                    "required_tool_category": "axe",
                }
            ],
        }
        cfg = EnvConfig(
            width=12,
            height=10,
            n_agents=1,
            max_steps=8,
            render_enabled=False,
            recipes_data=recipes_data,
        )
        env = PettingZooParallelRLRLGym(cfg)
        env.reset(seed=18)
        aid = "agent_0"
        pos = env.state.agents[aid].position
        env.state.stations[pos] = StationState(
            station_id="workbench",
            position=pos,
            speed_multiplier=1.0,
            quality_tier=0,
            unlock_recipes=["plank_test"],
        )
        env.state.agents[aid].inventory.append("log")
        _, _, _, _, info = env.step({aid: ACTION_INTERACT_STATION})
        self.assertIn("station_idle:workbench", info[aid]["events"])

        env.state.agents[aid].inventory.insert(0, "axe")
        env.step({aid: ACTION_EQUIP})
        _, _, _, _, info = env.step({aid: ACTION_INTERACT_STATION})
        self.assertTrue(any(str(evt).startswith("craft:plank_test:") for evt in info[aid]["events"]))
        self.assertGreaterEqual(env.state.agents[aid].inventory.count("wood_plank"), 2)

    def test_harvest_limit_blocks_tile_abuse(self):
        cfg = EnvConfig(width=12, height=10, n_agents=1, max_steps=5, render_enabled=False)
        env = PettingZooParallelRLRLGym(cfg)
        env.reset(seed=22)
        aid = "agent_0"
        env.state.agents[aid].position = (4, 4)
        env.state.grid[4][4] = "stone_floor"
        env.state.tile_harvest_counts[(4, 4)] = 12
        _, _, _, _, info = env.step({aid: ACTION_INTERACT_MINE})
        self.assertTrue(
            any(str(evt).startswith("harvest_tile_exhausted:stone_floor") for evt in info[aid]["events"])
        )

    def test_tool_durability_decreases_on_mining_use(self):
        cfg = EnvConfig(width=12, height=10, n_agents=1, max_steps=5, render_enabled=False)
        env = PettingZooParallelRLRLGym(cfg)
        env.reset(seed=23)
        aid = "agent_0"
        env.state.agents[aid].position = (4, 4)
        env.state.grid[4][4] = "stone_floor"
        env.state.agents[aid].inventory.insert(0, "stone_pickaxe")
        env.step({aid: ACTION_EQUIP})
        equipped = env.state.agents[aid].equipped[-1]
        before = env._item_current_durability(equipped)
        env.step({aid: ACTION_INTERACT_MINE})
        equipped_after = env.state.agents[aid].equipped[-1]
        after = env._item_current_durability(equipped_after)
        self.assertLess(after, before)

    def test_build_campfire_and_refuel(self):
        recipes_data = {
            "schema_version": 1,
            "recipes": [
                {
                    "id": "build_fire_test",
                    "inputs": {"campfire_kit": 1},
                    "outputs": {},
                    "skill": "crafting",
                    "station": "workbench",
                    "build_tile_id": "campfire",
                }
            ],
        }
        cfg = EnvConfig(
            width=12,
            height=10,
            n_agents=1,
            max_steps=8,
            render_enabled=False,
            recipes_data=recipes_data,
        )
        env = PettingZooParallelRLRLGym(cfg)
        env.reset(seed=24)
        aid = "agent_0"
        pos = env.state.agents[aid].position
        env.state.stations[pos] = StationState(
            station_id="workbench",
            position=pos,
            speed_multiplier=1.0,
            quality_tier=0,
            unlock_recipes=["build_fire_test"],
        )
        env.state.agents[aid].inventory.append("campfire_kit")
        env.step({aid: ACTION_INTERACT_STATION})
        fire_pos = None
        for nr, nc in ((pos[0] - 1, pos[1]), (pos[0] + 1, pos[1]), (pos[0], pos[1] - 1), (pos[0], pos[1] + 1)):
            if env.state.grid[nr][nc] == "campfire":
                fire_pos = (nr, nc)
                break
        self.assertIsNotNone(fire_pos)
        env.state.agents[aid].position = fire_pos
        env.state.agents[aid].inventory.append("stick")
        _, _, _, _, info = env.step({aid: ACTION_INTERACT})
        self.assertTrue(any(str(evt).startswith("fire_refuel:stick:") for evt in info[aid]["events"]))

    def test_spike_trap_damages_on_move(self):
        cfg = EnvConfig(width=12, height=10, n_agents=1, max_steps=5, render_enabled=False)
        env = PettingZooParallelRLRLGym(cfg)
        env.reset(seed=25)
        aid = "agent_0"
        env.state.agents[aid].position = (5, 5)
        env.state.grid[5][6] = "spike_trap"
        hp_before = env.state.agents[aid].hp
        env.step({aid: ACTION_MOVE_EAST})
        self.assertLess(env.state.agents[aid].hp, hp_before)


if __name__ == "__main__":
    unittest.main()
