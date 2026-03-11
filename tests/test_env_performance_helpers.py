import unittest

from rlrlgym.world.env import FIRE_CONTAINER_TILE_IDS, EnvConfig, MultiAgentRLRLGym
from rlrlgym.systems.models import AnimalState, ChestState, MonsterState


class TestEnvPerformanceHelpers(unittest.TestCase):
    def test_walkable_for_monster_uses_runtime_occupancy(self):
        env = MultiAgentRLRLGym(
            EnvConfig(width=8, height=8, n_agents=2, max_steps=5, render_enabled=False)
        )
        env.reset(seed=3)
        assert env.state is not None

        env.state.agents["agent_0"].position = (2, 2)
        env.state.agents["agent_1"].position = (2, 3)
        env.state.monsters = {
            "monster_0": MonsterState(
                entity_id="monster_0",
                monster_id="rat",
                name="Rat",
                symbol="r",
                color="red",
                position=(3, 3),
                hp=2,
                max_hp=2,
                acc=0,
                eva=0,
                dmg_min=1,
                dmg_max=1,
                dr_min=0,
                dr_max=0,
                alive=True,
            )
        }
        env.state.animals = {
            "animal_0": AnimalState(
                entity_id="animal_0",
                animal_id="rabbit",
                name="Rabbit",
                symbol="r",
                color="white",
                position=(4, 4),
                hp=2,
                max_hp=2,
                hunger=4,
                max_hunger=4,
                thirst=4,
                max_thirst=4,
                age=0,
                mature_age=4,
                reproduction_cooldown=0,
                reproduction_cooldown_max=4,
                alive=True,
            )
        }
        env._rebuild_runtime_caches()

        self.assertFalse(env._walkable_for_monster(2, 2, "monster_0"))
        self.assertFalse(env._walkable_for_monster(2, 3, "monster_0"))
        self.assertFalse(env._walkable_for_monster(4, 4, "monster_0"))
        self.assertTrue(env._walkable_for_monster(3, 3, "monster_0"))

    def test_nearest_food_distance_uses_cached_food_positions(self):
        env = MultiAgentRLRLGym(
            EnvConfig(width=8, height=8, n_agents=1, max_steps=5, render_enabled=False)
        )
        env.reset(seed=5)
        assert env.state is not None

        env.state.ground_items.clear()
        env.state.chests.clear()
        env._rebuild_runtime_caches()

        env._add_ground_item((1, 1), "fruit")
        env.state.chests[(6, 6)] = ChestState(position=(6, 6), opened=False, loot=["fruit"])
        env._refresh_chest_position((6, 6))

        self.assertEqual(env._nearest_food_distance((0, 0)), 2)
        self.assertEqual(env._nearest_food_distance((7, 7)), 2)

    def test_tick_fires_uses_active_fire_positions(self):
        env = MultiAgentRLRLGym(
            EnvConfig(width=8, height=8, n_agents=1, max_steps=5, render_enabled=False)
        )
        env.reset(seed=7)
        assert env.state is not None

        fire_tile = next(tile_id for tile_id in FIRE_CONTAINER_TILE_IDS if tile_id in env.tiles)
        env._set_grid_tile((2, 2), fire_tile)
        env._set_tile_interaction((2, 2), 1)
        env._active_fire_positions = {(2, 2)}

        env._tick_fires()

        self.assertNotIn((2, 2), env._active_fire_positions)
        self.assertNotIn((2, 2), env.state.tile_interactions)


if __name__ == "__main__":
    unittest.main()
