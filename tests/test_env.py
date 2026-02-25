import json
import tempfile
import unittest
from pathlib import Path

from rlrlgym import EnvConfig, PettingZooParallelRLRLGym
from rlrlgym.constants import (
    ACTION_ATTACK,
    ACTION_INTERACT,
    ACTION_LOOT,
    ACTION_MOVE_NORTH,
    ACTION_WAIT,
)
from rlrlgym.models import MonsterState


class TestEnv(unittest.TestCase):
    def test_reset_and_basic_step_shapes(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=2, max_steps=5)
        )
        obs, info = env.reset(seed=1)
        self.assertEqual(set(obs.keys()), {"agent_0", "agent_1"})
        self.assertTrue(info["agent_0"]["alive"])

        next_obs, rewards, terms, truncs, step_info = env.step(
            {"agent_0": ACTION_WAIT, "agent_1": ACTION_WAIT}
        )
        self.assertIn("agent_0", rewards)
        self.assertIn("agent_1", rewards)
        self.assertFalse(terms["agent_0"])
        self.assertFalse(truncs["agent_0"])
        self.assertIn("events", step_info["agent_0"])
        self.assertIn("agent_0", next_obs)

    def test_wall_blocks_movement(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=10, height=8, n_agents=1, max_steps=5)
        )
        env.reset(seed=2)
        pos = env.state.agents["agent_0"].position

        collision_seen = False
        for _ in range(10):
            _, rewards, _, _, info = env.step({"agent_0": ACTION_MOVE_NORTH})
            if "bump" in info["agent_0"]["events"]:
                collision_seen = True
                self.assertLess(rewards["agent_0"], 0)
                break

        self.assertTrue(collision_seen)
        self.assertEqual(env.state.agents["agent_0"].position[1], pos[1])

    def test_rendering_can_be_disabled(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=10, height=8, n_agents=1, render_enabled=False)
        )
        env.reset(seed=3)
        self.assertIsNone(env.render())

    def test_attack_can_attack_adjacent_agent(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=2, max_steps=20, render_enabled=False)
        )
        env.reset(seed=5)
        env.state.agents["agent_0"].position = (4, 4)
        env.state.agents["agent_1"].position = (4, 5)
        env.state.agents["agent_0"].equipped.append("dagger")
        starting_hp = env.state.agents["agent_1"].hp

        attack_seen = False
        damaged_seen = False
        for _ in range(8):
            _, _, _, _, info = env.step(
                {"agent_0": ACTION_ATTACK, "agent_1": ACTION_WAIT}
            )
            events = info["agent_0"]["events"]
            if any(e.startswith("agent_interact:attack:") for e in events):
                attack_seen = True
            if env.state.agents["agent_1"].hp < starting_hp:
                damaged_seen = True
                break

        self.assertTrue(attack_seen)
        self.assertTrue(damaged_seen)

    def test_interact_no_longer_attacks_adjacent_agent(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=2, max_steps=5, render_enabled=False)
        )
        env.reset(seed=31)
        env.state.agents["agent_0"].position = (4, 4)
        env.state.agents["agent_1"].position = (4, 5)
        env.state.agents["agent_0"].equipped.append("dagger")
        starting_hp = env.state.agents["agent_1"].hp
        _, _, _, _, info = env.step({"agent_0": ACTION_INTERACT, "agent_1": ACTION_WAIT})
        self.assertEqual(env.state.agents["agent_1"].hp, starting_hp)
        self.assertFalse(
            any(e.startswith("agent_interact:attack:") for e in info["agent_0"]["events"])
        )

    def test_chest_entity_can_be_opened(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=1, max_steps=10, render_enabled=False)
        )
        env.reset(seed=11)
        chest_pos = next(iter(env.state.chests.keys()))
        env.state.agents["agent_0"].position = chest_pos
        chest = env.state.chests[chest_pos]
        self.assertFalse(chest.opened)

        _, _, _, _, info = env.step({"agent_0": ACTION_LOOT})
        self.assertTrue(chest.opened)
        self.assertTrue(
            any(evt.startswith("chest_open:") for evt in info["agent_0"]["events"])
        )

    def test_class_starting_items_and_skill_modifiers(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(
                width=12,
                height=10,
                n_agents=1,
                max_steps=5,
                render_enabled=False,
                agent_profile_map={"agent_0": "human"},
                agent_class_map={"agent_0": "scout"},
            )
        )
        obs, info = env.reset(seed=13)
        a0 = env.state.agents["agent_0"]
        self.assertEqual(a0.class_name, "scout")
        self.assertIn("ration", a0.inventory)
        self.assertIn("torch", a0.inventory)
        self.assertGreaterEqual(a0.skills.get("exploration", 0), 2)
        self.assertEqual(obs["agent_0"]["class"], "scout")
        self.assertEqual(info["agent_0"]["class"], "scout")

    def test_custom_race_config_applies_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            races = {
                "schema_version": 1,
                "races": [
                    {
                        "name": "elf",
                        "strength": 4,
                        "dexterity": 9,
                        "intellect": 7,
                        "base_dr_min": 0,
                        "base_dr_max": 1,
                        "dr_bonus_vs": {"slash": 0, "pierce": 1, "blunt": -1},
                    },
                    {
                        "name": "human",
                        "strength": 5,
                        "dexterity": 6,
                        "intellect": 5,
                        "base_dr_min": 0,
                        "base_dr_max": 1,
                        "dr_bonus_vs": {"slash": 0, "pierce": 0, "blunt": 0},
                    },
                ],
            }
            race_path = Path(tmp) / "agent_races.json"
            race_path.write_text(json.dumps(races), encoding="utf-8")

            env = PettingZooParallelRLRLGym(
                EnvConfig(
                    width=12,
                    height=10,
                    n_agents=1,
                    render_enabled=False,
                    races_path=str(race_path),
                    agent_race_map={"agent_0": "elf"},
                    agent_class_map={"agent_0": "wanderer"},
                )
            )
            obs, info = env.reset(seed=17)
            a0 = env.state.agents["agent_0"]
            self.assertEqual(a0.race_name, "elf")
            self.assertEqual(a0.strength, 4)
            self.assertEqual(a0.dexterity, 9)
            self.assertEqual(a0.intellect, 7)
            self.assertEqual(obs["agent_0"]["race"], "elf")
            self.assertEqual(info["agent_0"]["race"], "elf")

    def test_skill_level_curve_and_overall_level(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(
                width=12,
                height=10,
                n_agents=1,
                render_enabled=False,
                agent_class_map={"agent_0": "wanderer"},
            )
        )
        env.reset(seed=19)
        a0 = env.state.agents["agent_0"]
        self.assertEqual(a0.skills.get("exploration", 0), 0)
        self.assertEqual(env._skill_xp_to_next(0), 20)
        self.assertEqual(env._skill_xp_to_next(1), 35)
        self.assertEqual(env._skill_xp_to_next(2), 50)

        events = []
        env._gain_skill_xp(a0, "exploration", 20, events)
        self.assertEqual(a0.skills.get("exploration", 0), 1)
        self.assertEqual(a0.skill_xp.get("exploration", 0), 0)
        env._gain_skill_xp(a0, "athletics", 55, events)
        self.assertEqual(a0.skills.get("athletics", 0), 2)
        self.assertEqual(a0.skill_xp.get("athletics", 0), 0)
        self.assertEqual(env._overall_level(a0), 3)

    def test_monster_attacks_when_adjacent(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=1, max_steps=5, render_enabled=False)
        )
        env.reset(seed=23)
        env.state.agents["agent_0"].position = (4, 4)
        env.state.monsters = {
            "monster_0": MonsterState(
                entity_id="monster_0",
                monster_id="rat",
                name="Rat",
                symbol="r",
                color="yellow",
                position=(4, 5),
                hp=3,
                max_hp=3,
                acc=10,
                eva=0,
                dmg_min=2,
                dmg_max=2,
                dr_min=0,
                dr_max=0,
                alive=True,
            )
        }
        start_hp = env.state.agents["agent_0"].hp
        _, _, _, _, info = env.step({"agent_0": ACTION_WAIT})
        events = info["agent_0"]["events"]
        self.assertTrue(
            any(evt.startswith("monster_attack:") for evt in events),
            msg=str(events),
        )
        self.assertLess(env.state.agents["agent_0"].hp, start_hp)

    def test_monster_moves_toward_agent_when_not_adjacent(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=1, max_steps=5, render_enabled=False)
        )
        env.reset(seed=29)
        env.state.grid = [
            ["floor" for _ in row]
            for row in env.state.grid
        ]
        env.state.agents["agent_0"].position = (4, 7)
        env.state.monsters = {
            "monster_0": MonsterState(
                entity_id="monster_0",
                monster_id="rat",
                name="Rat",
                symbol="r",
                color="yellow",
                position=(4, 2),
                hp=3,
                max_hp=3,
                acc=0,
                eva=0,
                dmg_min=1,
                dmg_max=1,
                dr_min=0,
                dr_max=0,
                alive=True,
            )
        }
        before = env._manhattan(
            env.state.monsters["monster_0"].position, env.state.agents["agent_0"].position
        )
        env.step({"agent_0": ACTION_WAIT})
        after = env._manhattan(
            env.state.monsters["monster_0"].position, env.state.agents["agent_0"].position
        )
        self.assertLess(after, before)


if __name__ == "__main__":
    unittest.main()
