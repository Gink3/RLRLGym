import unittest

from rlrlgym.content.env_constants import (
    load_combat_constants,
    load_reward_constants,
    load_world_constants,
)


class TestEnvConstants(unittest.TestCase):
    def test_reward_constants_load_from_json(self):
        constants = load_reward_constants()
        self.assertAlmostEqual(constants.move_valid_reward, 0.005)
        self.assertAlmostEqual(constants.eat_waste_threshold, 0.8)

    def test_combat_constants_load_from_json(self):
        constants = load_combat_constants()
        self.assertEqual(constants.damage_type_blunt, "blunt")
        self.assertEqual(constants.unarmed_damage_range, (1, 2))
        self.assertIn(("head", 14), constants.hit_slot_weights)
        self.assertEqual(constants.hit_slot_to_armor_slots["rings"], ("ring_1", "ring_2", "ring_3", "ring_4"))

    def test_world_constants_load_from_json(self):
        constants = load_world_constants()
        self.assertEqual(constants.default_vision_range, 20)
        self.assertIn("campfire", constants.fire_container_tile_ids)
        self.assertIn("wood_wall", constants.default_construct_tile_ids)
        self.assertIn("tree", constants.opaque_tile_ids)


if __name__ == "__main__":
    unittest.main()
