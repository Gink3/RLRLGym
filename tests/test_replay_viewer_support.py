import unittest

from rlrlgym.content.tiles import load_tileset
from rlrlgym.world.replay_viewer_support import (
    build_activity_log_lines,
    resource_node_sprite_id,
    supported_animal_sprite_ids,
    supported_construction_sprite_ids,
    supported_resource_sprite_ids,
    visible_tiles_for_agent,
)


class TestReplayViewerSupport(unittest.TestCase):
    def test_sprite_catalogs_cover_requested_entities(self):
        animals = supported_animal_sprite_ids()
        constructions = supported_construction_sprite_ids()
        resources = supported_resource_sprite_ids()

        self.assertIn("rabbit", animals)
        self.assertIn("sheep", animals)
        self.assertIn("workbench", constructions)
        self.assertIn("wood_wall", constructions)
        self.assertIn("clay_patch", resources)
        self.assertIn("tree", resources)

    def test_resource_node_sprite_mapping_groups_common_nodes(self):
        self.assertEqual(resource_node_sprite_id("timber_pine", "wood"), "tree")
        self.assertEqual(resource_node_sprite_id("berry_patch", "berries"), "berry_bush")
        self.assertEqual(resource_node_sprite_id("clay_deposit", "clay"), "clay_patch")
        self.assertEqual(resource_node_sprite_id("copper_vein", "copper_ore"), "ore_vein")

    def test_activity_log_filter_keeps_actor_and_target_events(self):
        step = {
            "agents": {
                "agent_0": {
                    "action": 10,
                    "reward": 1.5,
                    "events": ["team_give:agent_1:ration"],
                    "terminated": False,
                    "truncated": False,
                    "death_reason": None,
                    "winner": False,
                },
                "agent_1": {
                    "action": 4,
                    "reward": 0.2,
                    "events": ["enemy_visible"],
                    "terminated": False,
                    "truncated": False,
                    "death_reason": None,
                    "winner": False,
                },
            },
            "agent_damage": [
                {"agent_id": "agent_1", "amount": 3, "source": "agent:agent_0"},
            ],
        }
        lines = build_activity_log_lines(step, {"agent_1"})

        joined = "\n".join(lines)
        self.assertIn("agent_0: action=10 reward=1.500", joined)
        self.assertIn("team_give:agent_1:ration", joined)
        self.assertIn("agent_1: action=4 reward=0.200", joined)
        self.assertIn("damage: agent_1 <- agent:agent_0 (3)", joined)

    def test_visible_tiles_uses_line_of_sight(self):
        tiles = load_tileset("data/base/tiles.json")
        frame = {
            "grid": [
                ["floor", "floor", "floor"],
                ["floor", "wall", "floor"],
                ["floor", "floor", "floor"],
            ],
            "agents": {
                "agent_0": {
                    "position": [2, 0],
                    "alive": True,
                    "profile_name": "human",
                    "skills": {"exploration": 0},
                }
            },
        }

        visible = visible_tiles_for_agent(frame, "agent_0", tile_defs=tiles)

        self.assertIn((2, 0), visible)
        self.assertIn((1, 0), visible)
        self.assertNotIn((0, 2), visible)


if __name__ == "__main__":
    unittest.main()
