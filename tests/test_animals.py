import unittest

from rlrlgym import EnvConfig, PettingZooParallelRLRLGym
from rlrlgym.constants import ACTION_INTERACT, ACTION_WAIT
from rlrlgym.models import AnimalState


class TestAnimals(unittest.TestCase):
    def test_animals_spawn_when_density_enabled(self):
        cfg = EnvConfig(
            width=14,
            height=10,
            n_agents=1,
            max_steps=5,
            render_enabled=False,
            mapgen_config_data={
                "schema_version": 1,
                "mapgen": {
                    "wall_tile_id": "wall",
                    "floor_fallback_id": "floor",
                    "chest_density": 0.0,
                    "monster_density": 0.0,
                    "animal_density": 0.06,
                    "min_width": 4,
                    "min_height": 4,
                    "biomes": [
                        {
                            "id": "field",
                            "weight": 1.0,
                            "tile_weights": {"floor": 0.6, "grass": 0.4},
                        }
                    ],
                },
            },
        )
        env = PettingZooParallelRLRLGym(cfg)
        env.reset(seed=101)
        self.assertGreater(len(env.state.animals), 0)

    def test_sheep_can_be_sheared(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=1, max_steps=5, render_enabled=False)
        )
        env.reset(seed=102)
        aid = "agent_0"
        agent = env.state.agents[aid]
        agent.position = (5, 5)
        env.state.grid[5][5] = "floor"
        env.state.resource_nodes = {}
        env.state.stations = {}
        env.state.chests = {}
        env.state.animals = {
            "animal_test_sheep": AnimalState(
                entity_id="animal_test_sheep",
                animal_id="sheep",
                name="Sheep",
                symbol="S",
                color="white",
                position=(5, 5),
                hp=8,
                max_hp=8,
                hunger=8,
                max_hunger=12,
                thirst=8,
                max_thirst=12,
                age=9,
                mature_age=8,
                reproduction_cooldown=0,
                reproduction_cooldown_max=8,
                can_shear=True,
                sheared=False,
                shear_item="sheep_wool",
                wool_regrow=0,
                shear_regrow_max=6,
                alive=True,
            )
        }
        _, _, _, _, info = env.step({aid: ACTION_INTERACT})
        self.assertIn("sheep_wool", env.state.agents[aid].inventory)
        self.assertTrue(
            any(str(evt).startswith("shear:sheep:") for evt in info[aid]["events"])
        )

    def test_animal_starves_without_food(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=1, max_steps=5, render_enabled=False)
        )
        env.reset(seed=103)
        for r in range(1, env.config.height - 1):
            for c in range(1, env.config.width - 1):
                env.state.grid[r][c] = "floor"
        env.state.animals = {
            "animal_starve": AnimalState(
                entity_id="animal_starve",
                animal_id="pig",
                name="Pig",
                symbol="P",
                color="yellow",
                position=(5, 5),
                hp=9,
                max_hp=9,
                hunger=1,
                max_hunger=13,
                thirst=5,
                max_thirst=13,
                age=9,
                mature_age=8,
                reproduction_cooldown=0,
                reproduction_cooldown_max=9,
                can_shear=False,
                sheared=False,
                shear_item="",
                wool_regrow=0,
                shear_regrow_max=0,
                alive=True,
            )
        }
        env.step({"agent_0": ACTION_WAIT})
        pig = env.state.animals["animal_starve"]
        self.assertFalse(pig.alive)
        self.assertIn("pig_leather", env.state.ground_items.get(pig.position, []))

    def test_mature_animals_can_reproduce(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=1, max_steps=5, render_enabled=False)
        )
        env.reset(seed=104)
        env.state.animals = {
            "animal_cow_a": AnimalState(
                entity_id="animal_cow_a",
                animal_id="cow",
                name="Cow",
                symbol="C",
                color="white",
                position=(5, 5),
                hp=10,
                max_hp=10,
                hunger=10,
                max_hunger=14,
                thirst=10,
                max_thirst=14,
                age=12,
                mature_age=10,
                reproduction_cooldown=0,
                reproduction_cooldown_max=10,
                can_shear=False,
                sheared=False,
                shear_item="",
                wool_regrow=0,
                shear_regrow_max=0,
                alive=True,
            ),
            "animal_cow_b": AnimalState(
                entity_id="animal_cow_b",
                animal_id="cow",
                name="Cow",
                symbol="C",
                color="white",
                position=(5, 6),
                hp=10,
                max_hp=10,
                hunger=10,
                max_hunger=14,
                thirst=10,
                max_thirst=14,
                age=12,
                mature_age=10,
                reproduction_cooldown=0,
                reproduction_cooldown_max=10,
                can_shear=False,
                sheared=False,
                shear_item="",
                wool_regrow=0,
                shear_regrow_max=0,
                alive=True,
            ),
        }
        before = len(env.state.animals)
        env._animal_try_reproduce(env.state.animals["animal_cow_a"])
        self.assertGreater(len(env.state.animals), before)

    def test_animals_consume_forage_tiles(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=1, max_steps=5, render_enabled=False)
        )
        env.reset(seed=105)
        for nr, nc in ((4, 5), (6, 5), (5, 4), (5, 6)):
            env.state.grid[nr][nc] = "floor"
        env.state.grid[5][5] = "grass"
        animal = AnimalState(
            entity_id="animal_grazer",
            animal_id="sheep",
            name="Sheep",
            symbol="S",
            color="white",
            position=(5, 5),
            hp=8,
            max_hp=8,
            hunger=2,
            max_hunger=12,
            thirst=8,
            max_thirst=12,
            age=9,
            mature_age=8,
            reproduction_cooldown=3,
            reproduction_cooldown_max=8,
            can_shear=True,
            sheared=False,
            shear_item="sheep_wool",
            wool_regrow=0,
            shear_regrow_max=6,
            alive=True,
        )
        env._animal_tick_needs(animal)
        self.assertGreater(animal.hunger, 2)
        env._animal_tick_needs(animal)
        self.assertEqual(env.state.grid[5][5], "floor")


if __name__ == "__main__":
    unittest.main()
