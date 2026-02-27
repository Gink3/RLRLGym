import json
import tempfile
import unittest
from pathlib import Path

from rlrlgym import EnvConfig, PettingZooParallelRLRLGym
from rlrlgym.constants import (
    ACTION_ATTACK,
    ACTION_EQUIP,
    ACTION_GIVE,
    ACTION_GUARD,
    ACTION_INTERACT,
    ACTION_ACCEPT_INVITE,
    ACTION_LEAVE_FACTION,
    ACTION_LOOT,
    ACTION_MOVE_NORTH,
    ACTION_REVIVE,
    ACTION_TRADE,
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
                agent_class_map={"agent_0": "rogue"},
            )
        )
        obs, info = env.reset(seed=13)
        a0 = env.state.agents["agent_0"]
        self.assertEqual(a0.class_name, "rogue")
        self.assertIn("thrown_knife", a0.inventory)
        self.assertIn("torch", a0.inventory)
        self.assertGreaterEqual(a0.skills.get("thrown_weapons", 0), 2)
        self.assertEqual(obs["agent_0"]["class"], "rogue")
        self.assertEqual(info["agent_0"]["class"], "rogue")

    def test_raises_when_class_references_unknown_item(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_classes = {
                "schema_version": 1,
                "classes": [
                    {
                        "name": "fighter",
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
                    agent_class_map={"agent_0": "fighter"},
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
                agent_class_map={"agent_0": "fighter"},
            )
        )
        env.reset(seed=19)
        a0 = env.state.agents["agent_0"]
        start_athletics = int(a0.skills.get("athletics", 0))
        start_overall = int(env._overall_level(a0))
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
        self.assertGreaterEqual(a0.skills.get("athletics", 0), start_athletics + 1)
        self.assertGreaterEqual(a0.skill_xp.get("athletics", 0), 0)
        self.assertGreater(a0.hp, 1)
        self.assertGreater(env._overall_level(a0), start_overall)

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

    def test_interact_creates_faction_for_unaligned_agent(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=1, max_steps=5, render_enabled=False)
        )
        env.reset(seed=43)
        a0 = env.state.agents["agent_0"]
        self.assertEqual(a0.faction_id, -1)

        _, _, _, _, info = env.step({"agent_0": ACTION_INTERACT})
        self.assertGreaterEqual(a0.faction_id, 1)
        self.assertEqual(env.state.faction_leaders[a0.faction_id], "agent_0")
        self.assertTrue(
            any(evt.startswith("faction_create:") for evt in info["agent_0"]["events"])
        )

    def test_faction_invite_join_and_leader_share(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=2, max_steps=10, render_enabled=False)
        )
        env.reset(seed=47)
        env.state.agents["agent_0"].position = (4, 4)
        env.state.agents["agent_1"].position = (4, 5)

        env.step({"agent_0": ACTION_INTERACT, "agent_1": ACTION_WAIT})
        faction_id = int(env.state.agents["agent_0"].faction_id)
        self.assertGreaterEqual(faction_id, 1)

        _, _, _, _, invite_info = env.step(
            {"agent_0": ACTION_INTERACT, "agent_1": ACTION_WAIT}
        )
        self.assertEqual(
            int(env.state.pending_faction_invites["agent_1"]["faction_id"]),
            faction_id,
        )
        self.assertTrue(
            any(
                evt.startswith("faction_invite_sent:agent_1:")
                for evt in invite_info["agent_0"]["events"]
            )
        )

        _, rewards, _, _, join_info = env.step(
            {"agent_0": ACTION_WAIT, "agent_1": ACTION_INTERACT}
        )
        self.assertEqual(int(env.state.agents["agent_1"].faction_id), faction_id)
        self.assertNotIn("agent_1", env.state.pending_faction_invites)
        self.assertTrue(
            any(evt.startswith("faction_join:") for evt in join_info["agent_1"]["events"])
        )
        self.assertTrue(
            any(evt.startswith("leader_team_share:") for evt in join_info["agent_0"]["events"]),
            msg=str(join_info["agent_0"]["events"]),
        )
        self.assertGreater(rewards["agent_0"], -0.05)

    def test_join_awards_small_bonus_to_inviter_and_invitee_once(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(
                width=12,
                height=10,
                n_agents=2,
                max_steps=12,
                render_enabled=False,
                team_join_inviter_reward=0.05,
                team_join_invitee_reward=0.05,
            )
        )
        env.reset(seed=95)
        a0 = env.state.agents["agent_0"]
        a1 = env.state.agents["agent_1"]
        a0.position = (4, 4)
        a1.position = (4, 5)

        env.step({"agent_0": ACTION_INTERACT, "agent_1": ACTION_WAIT})
        env.step({"agent_0": ACTION_INTERACT, "agent_1": ACTION_WAIT})
        _, rewards, _, _, join_info = env.step(
            {"agent_0": ACTION_WAIT, "agent_1": ACTION_ACCEPT_INVITE}
        )
        self.assertTrue(
            any(
                evt.startswith("faction_join_inviter_bonus:agent_1:")
                for evt in join_info["agent_0"]["events"]
            )
        )
        self.assertGreaterEqual(rewards["agent_1"], 0.05)

        # Leave and rejoin should not grant join-side bonus again for same invitee.
        env.step({"agent_0": ACTION_WAIT, "agent_1": ACTION_LEAVE_FACTION})
        env.state.agents["agent_1"].position = (4, 5)
        env.step({"agent_0": ACTION_INTERACT, "agent_1": ACTION_WAIT})
        _, _, _, _, second_join = env.step(
            {"agent_0": ACTION_WAIT, "agent_1": ACTION_ACCEPT_INVITE}
        )
        self.assertFalse(
            any(
                evt.startswith("faction_join_inviter_bonus:agent_1:")
                for evt in second_join["agent_0"]["events"]
            )
        )

    def test_invite_ttl_expires_pending_invite(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(
                width=12,
                height=10,
                n_agents=2,
                max_steps=8,
                render_enabled=False,
                faction_invite_ttl_steps=1,
            )
        )
        env.reset(seed=97)
        env.state.agents["agent_0"].position = (4, 4)
        env.state.agents["agent_1"].position = (4, 5)
        env.step({"agent_0": ACTION_INTERACT, "agent_1": ACTION_WAIT})
        env.step({"agent_0": ACTION_INTERACT, "agent_1": ACTION_WAIT})
        self.assertIn("agent_1", env.state.pending_faction_invites)

        env.step({"agent_0": ACTION_WAIT, "agent_1": ACTION_WAIT})

        _, rewards, _, _, info = env.step(
            {"agent_0": ACTION_WAIT, "agent_1": ACTION_ACCEPT_INVITE}
        )
        self.assertLess(rewards["agent_1"], 0.0)
        self.assertNotIn("agent_1", env.state.pending_faction_invites)
        self.assertTrue(
            any(
                evt.startswith("faction_accept_fail:no_invite")
                for evt in info["agent_1"]["events"]
            )
        )

    def test_faction_action_cooldown_blocks_repeat_invites(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(
                width=12,
                height=10,
                n_agents=2,
                max_steps=8,
                render_enabled=False,
                faction_action_cooldown_steps=2,
            )
        )
        env.reset(seed=101)
        env.state.agents["agent_0"].position = (4, 4)
        env.state.agents["agent_1"].position = (4, 5)
        env.step({"agent_0": ACTION_INTERACT, "agent_1": ACTION_WAIT})
        env.step({"agent_0": ACTION_INTERACT, "agent_1": ACTION_WAIT})
        _, rewards, _, _, info = env.step(
            {"agent_0": ACTION_INTERACT, "agent_1": ACTION_WAIT}
        )
        self.assertLess(rewards["agent_0"], 0.0)
        self.assertTrue(
            any(
                evt.startswith("invite_faction_cooldown:")
                for evt in info["agent_0"]["events"]
            ),
            msg=str(info["agent_0"]["events"]),
        )

    def test_team_reward_guard_prevents_trade_loop_farming(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(
                width=12,
                height=10,
                n_agents=2,
                max_steps=8,
                render_enabled=False,
                faction_action_cooldown_steps=0,
                team_action_bonus=0.0,
                team_proximity_reward=0.0,
                team_pair_reward_cap_per_episode=1,
                team_pair_reward_repeat_guard_steps=10,
            )
        )
        env.reset(seed=103)
        a0 = env.state.agents["agent_0"]
        a1 = env.state.agents["agent_1"]
        a0.position = (4, 4)
        a1.position = (4, 5)
        a0.faction_id = 20
        a1.faction_id = 20
        a0.inventory = ["ration"]
        a1.inventory = ["fruit"]
        _, rewards1, _, _, _ = env.step({"agent_0": ACTION_TRADE, "agent_1": ACTION_WAIT})
        _, rewards2, _, _, info2 = env.step(
            {"agent_0": ACTION_TRADE, "agent_1": ACTION_WAIT}
        )
        self.assertTrue(
            any(
                evt.startswith("team_trade_reward_guard:")
                for evt in info2["agent_0"]["events"]
            )
        )
        self.assertLess(rewards2["agent_0"], rewards1["agent_0"])

    def test_leave_faction_sets_agent_to_solo_and_reassigns_leader(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=2, max_steps=8, render_enabled=False)
        )
        env.reset(seed=51)
        a0 = env.state.agents["agent_0"]
        a1 = env.state.agents["agent_1"]
        a0.faction_id = 3
        a1.faction_id = 3
        env.state.faction_leaders[3] = "agent_0"

        _, _, _, _, info = env.step(
            {"agent_0": ACTION_LEAVE_FACTION, "agent_1": ACTION_WAIT}
        )
        self.assertEqual(a0.faction_id, -1)
        self.assertEqual(env.state.faction_leaders.get(3), "agent_1")
        self.assertTrue(
            any(evt.startswith("faction_leave:3") for evt in info["agent_0"]["events"])
        )

    def test_leave_faction_without_faction_is_penalized(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=1, max_steps=5, render_enabled=False)
        )
        env.reset(seed=91)
        _, rewards, _, _, info = env.step({"agent_0": ACTION_LEAVE_FACTION})
        self.assertLess(rewards["agent_0"], 0.0)
        self.assertTrue(
            any(
                evt.startswith("faction_leave_fail:solo")
                for evt in info["agent_0"]["events"]
            )
        )

    def test_accept_invite_without_pending_invite_is_penalized(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=1, max_steps=5, render_enabled=False)
        )
        env.reset(seed=93)
        _, rewards, _, _, info = env.step({"agent_0": ACTION_ACCEPT_INVITE})
        self.assertLess(rewards["agent_0"], 0.0)
        self.assertTrue(
            any(
                evt.startswith("faction_accept_fail:no_invite")
                for evt in info["agent_0"]["events"]
            )
        )

    def test_treasure_hold_reward_and_end_bonus(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(
                width=12,
                height=10,
                n_agents=1,
                max_steps=1,
                render_enabled=False,
                treasure_hold_reward_per_turn=0.02,
                treasure_end_bonus_per_item=0.5,
            )
        )
        env.reset(seed=57)
        a0 = env.state.agents["agent_0"]
        a0.inventory = ["coin", "ring_of_guarding"]
        _, rewards, _, truncs, info = env.step({"agent_0": ACTION_WAIT})
        self.assertTrue(truncs["agent_0"])
        self.assertTrue(
            any(evt.startswith("treasure_hold:2") for evt in info["agent_0"]["events"])
        )
        self.assertTrue(
            any(evt.startswith("treasure_end_bonus:2") for evt in info["agent_0"]["events"])
        )
        components = info["agent_0"].get("reward_components", {})
        self.assertGreaterEqual(float(components.get("treasure", 0.0)), 1.0)
        self.assertGreater(rewards["agent_0"], 0.75)

    def test_nearby_agent_counts_include_relation_status(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=3, max_steps=5, render_enabled=False)
        )
        env.reset(seed=63)
        a0 = env.state.agents["agent_0"]
        a1 = env.state.agents["agent_1"]
        a2 = env.state.agents["agent_2"]
        a0.position = (4, 4)
        a1.position = (4, 5)
        a2.position = (5, 4)
        a0.faction_id = 5
        a1.faction_id = 5
        a2.faction_id = 6
        obs = env._build_observation("agent_0")
        rel = obs["stats"]["nearby_agents"]["relation_counts"]
        self.assertEqual(rel.get("ally"), 1)
        self.assertEqual(rel.get("enemy"), 1)

    def test_attacking_allied_agent_applies_penalty(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=2, max_steps=8, render_enabled=False)
        )
        env.reset(seed=53)
        a0 = env.state.agents["agent_0"]
        a1 = env.state.agents["agent_1"]
        a0.position = (4, 4)
        a1.position = (4, 5)
        a0.faction_id = 1
        a1.faction_id = 1
        a0.equipped.append("dagger")

        penalty_seen = False
        for _ in range(10):
            _, rewards, _, _, info = env.step(
                {"agent_0": ACTION_ATTACK, "agent_1": ACTION_WAIT}
            )
            events = info["agent_0"]["events"]
            if any(evt.startswith("ally_damage_penalty:") for evt in events):
                penalty_seen = True
                components = info["agent_0"].get("reward_components", {})
                self.assertLess(float(components.get("action_total", 0.0)), 0.0)
                break
        self.assertTrue(penalty_seen)

    def test_give_action_transfers_item_to_adjacent_ally(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=2, max_steps=5, render_enabled=False)
        )
        env.reset(seed=59)
        a0 = env.state.agents["agent_0"]
        a1 = env.state.agents["agent_1"]
        a0.position = (4, 4)
        a1.position = (4, 5)
        a0.faction_id = 7
        a1.faction_id = 7
        a0.inventory = ["ration"]
        a1.inventory = []

        _, _, _, _, info = env.step({"agent_0": ACTION_GIVE, "agent_1": ACTION_WAIT})
        self.assertEqual(a0.inventory, [])
        self.assertIn("ration", a1.inventory)
        self.assertTrue(
            any(evt.startswith("team_give:agent_1:ration") for evt in info["agent_0"]["events"])
        )

    def test_trade_action_swaps_first_item_with_adjacent_ally(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=2, max_steps=5, render_enabled=False)
        )
        env.reset(seed=61)
        a0 = env.state.agents["agent_0"]
        a1 = env.state.agents["agent_1"]
        a0.position = (4, 4)
        a1.position = (4, 5)
        a0.faction_id = 9
        a1.faction_id = 9
        a0.inventory = ["ration"]
        a1.inventory = ["fruit"]

        _, _, _, _, info = env.step({"agent_0": ACTION_TRADE, "agent_1": ACTION_WAIT})
        self.assertEqual(a0.inventory[0], "fruit")
        self.assertEqual(a1.inventory[0], "ration")
        self.assertTrue(
            any(evt.startswith("team_trade:agent_1:ration<->fruit") for evt in info["agent_0"]["events"])
        )

    def test_revive_action_restores_adjacent_dead_ally(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=2, max_steps=6, render_enabled=False)
        )
        env.reset(seed=67)
        a0 = env.state.agents["agent_0"]
        a1 = env.state.agents["agent_1"]
        a0.position = (4, 4)
        a1.position = (4, 5)
        a0.faction_id = 11
        a1.faction_id = 11
        a0.inventory = ["bandage"]
        a1.alive = False
        a1.hp = 0

        _, _, terms, _, info = env.step({"agent_0": ACTION_REVIVE, "agent_1": ACTION_WAIT})
        self.assertTrue(a1.alive)
        self.assertGreater(a1.hp, 0)
        self.assertFalse(terms["agent_1"])
        self.assertTrue(
            any(evt.startswith("team_revive:agent_1:bandage") for evt in info["agent_0"]["events"])
        )

    def test_guard_action_reduces_damage_taken(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=2, max_steps=6, render_enabled=False)
        )
        env.reset(seed=71)
        a0 = env.state.agents["agent_0"]
        a1 = env.state.agents["agent_1"]
        a0.position = (4, 4)
        a1.position = (4, 5)
        a0.faction_id = 12
        a1.faction_id = 12
        a0.hp = 30
        a1.hp = 30
        env.state.monsters = {
            "monster_0": MonsterState(
                entity_id="monster_0",
                monster_id="rat",
                name="Rat",
                symbol="r",
                color="yellow",
                position=(4, 6),
                hp=3,
                max_hp=3,
                acc=12,
                eva=0,
                dmg_min=6,
                dmg_max=6,
                dr_min=0,
                dr_max=0,
                alive=True,
            )
        }
        _, _, _, _, info = env.step({"agent_0": ACTION_GUARD, "agent_1": ACTION_WAIT})
        self.assertTrue(
            any(evt.startswith("team_guard:agent_1") for evt in info["agent_0"]["events"])
        )
        self.assertTrue(
            any(
                evt.startswith("team_guard_block:agent_0:agent_1:")
                for evt in info["agent_1"]["events"]
            ),
            msg=str(info["agent_1"]["events"]),
        )

    def test_formation_bonus_increases_damage_reduction(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=2, max_steps=5, render_enabled=False)
        )
        env.reset(seed=73)
        a0 = env.state.agents["agent_0"]
        a1 = env.state.agents["agent_1"]
        a0.position = (4, 4)
        a1.position = (4, 5)
        a0.faction_id = 13
        a1.faction_id = 13
        solo_dr, _, _, _ = env._roll_hit_location_dr(a0, DAMAGE_TYPE_SLASH, forced_hit_slot="chest")
        a1.position = (8, 8)
        spread_dr, _, _, _ = env._roll_hit_location_dr(a0, DAMAGE_TYPE_SLASH, forced_hit_slot="chest")
        self.assertGreaterEqual(solo_dr, spread_dr + 1)

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
