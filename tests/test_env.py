import json
import tempfile
import unittest
from pathlib import Path

from rlrlgym import EnvConfig, PettingZooParallelRLRLGym
from rlrlgym.constants import (
    ACTION_ATTACK,
    ACTION_EQUIP,
    ACTION_INTERACT,
    ACTION_LOOT,
    ACTION_MOVE_NORTH,
    ACTION_WAIT,
)
from rlrlgym.env import DAMAGE_TYPE_BLUNT, DAMAGE_TYPE_PIERCE, DAMAGE_TYPE_SLASH
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

    def test_raises_when_class_references_unknown_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_classes = {
                "schema_version": 1,
                "classes": [
                    {
                        "name": "wanderer",
                        "starting_items": ["missing_item"],
                        "skill_modifiers": {},
                    }
                ],
            }
            classes_path = Path(tmp) / "agent_classes.json"
            classes_path.write_text(json.dumps(bad_classes), encoding="utf-8")

            with self.assertRaises(ValueError):
                PettingZooParallelRLRLGym(
                    EnvConfig(
                        width=12,
                        height=10,
                        n_agents=1,
                        render_enabled=False,
                        classes_path=str(classes_path),
                    )
                )

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
        a0.hp = 2
        env._gain_skill_xp(a0, "exploration", 20, events)
        self.assertEqual(a0.skills.get("exploration", 0), 1)
        self.assertEqual(a0.skill_xp.get("exploration", 0), 0)
        self.assertEqual(a0.hp, 7)
        a0.hp = 1
        env._gain_skill_xp(a0, "athletics", 55, events)
        self.assertEqual(a0.skills.get("athletics", 0), 2)
        self.assertEqual(a0.skill_xp.get("athletics", 0), 0)
        self.assertEqual(a0.hp, a0.max_hp)
        self.assertEqual(env._overall_level(a0), 3)

    def test_equip_armor_replaces_existing_slot_item(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=1, max_steps=5, render_enabled=False)
        )
        env.reset(seed=37)
        a0 = env.state.agents["agent_0"]
        a0.inventory = ["leather_cap", "bronze_helm"]

        env.step({"agent_0": ACTION_EQUIP})
        self.assertIn("leather_cap", a0.equipped)
        self.assertNotIn("leather_cap", a0.inventory)
        self.assertEqual(a0.armor_slots.get("head"), "leather_cap")

        _, _, _, _, info = env.step({"agent_0": ACTION_EQUIP})
        self.assertIn("bronze_helm", a0.equipped)
        self.assertNotIn("leather_cap", a0.equipped)
        self.assertIn("leather_cap", a0.inventory)
        self.assertEqual(a0.armor_slots.get("head"), "bronze_helm")
        self.assertTrue(
            any(evt.startswith("unequip:head:leather_cap") for evt in info["agent_0"]["events"])
        )

    def test_armor_slots_add_damage_reduction(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=1, max_steps=5, render_enabled=False)
        )
        env.reset(seed=41)
        a0 = env.state.agents["agent_0"]
        a0.equipped = ["steel_plate", "chain_mantle", "leather_cap", "guardian_torc"]
        a0.armor_slots["chest"] = "steel_plate"
        a0.armor_slots["back"] = "chain_mantle"
        a0.armor_slots["head"] = "leather_cap"
        a0.armor_slots["neck"] = "guardian_torc"

        chest_dr, chest_hit_slot, _, _ = env._roll_hit_location_dr(
            a0, DAMAGE_TYPE_SLASH, forced_hit_slot="chest"
        )
        back_dr, back_hit_slot, _, _ = env._roll_hit_location_dr(
            a0, DAMAGE_TYPE_PIERCE, forced_hit_slot="back"
        )
        neck_dr, neck_hit_slot, _, _ = env._roll_hit_location_dr(
            a0, DAMAGE_TYPE_BLUNT, forced_hit_slot="neck"
        )
        self.assertEqual(chest_hit_slot, "chest")
        self.assertEqual(back_hit_slot, "back")
        self.assertEqual(neck_hit_slot, "neck")
        self.assertGreaterEqual(chest_dr, 4)
        self.assertGreaterEqual(back_dr, 1)
        self.assertGreaterEqual(neck_dr, 2)

    def test_ring_items_fill_ring_slots_in_order(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=1, max_steps=8, render_enabled=False)
        )
        env.reset(seed=43)
        a0 = env.state.agents["agent_0"]
        a0.inventory = [
            "ring_of_guarding",
            "ring_of_blades",
            "ring_of_warding",
            "ring_of_bastion",
            "ring_of_balance",
        ]

        for _ in range(5):
            env.step({"agent_0": ACTION_EQUIP})

        self.assertEqual(a0.armor_slots.get("ring_1"), "ring_of_balance")
        self.assertEqual(a0.armor_slots.get("ring_2"), "ring_of_blades")
        self.assertEqual(a0.armor_slots.get("ring_3"), "ring_of_warding")
        self.assertEqual(a0.armor_slots.get("ring_4"), "ring_of_bastion")
        self.assertIn("ring_of_guarding", a0.inventory)

    def test_armor_skill_xp_gained_when_hit_location_armor_mitigates(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=2, max_steps=12, render_enabled=False)
        )
        env.reset(seed=47)
        attacker = env.state.agents["agent_0"]
        defender = env.state.agents["agent_1"]
        attacker.position = (4, 4)
        defender.position = (4, 5)
        attacker.dexterity = 20
        attacker.equipped.append("dagger")
        defender.hp = 100
        defender.max_hp = 100
        defender.armor_slots["head"] = "steel_helm"
        defender.equipped.append("steel_helm")

        start_xp = int(defender.skill_xp.get("armor_head", 0))
        for _ in range(20):
            env._attack_agent(attacker, defender, events=[])
            if int(defender.skill_xp.get("armor_head", 0)) > start_xp:
                break
        self.assertGreater(int(defender.skill_xp.get("armor_head", 0)), start_xp)

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
