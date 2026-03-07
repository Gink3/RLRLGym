import unittest

from rlrlgym import EnvConfig, PettingZooParallelRLRLGym
from rlrlgym.systems.constants import ACTION_GIVE, ACTION_USE, ACTION_WAIT


class TestStatusSpellBookEnchant(unittest.TestCase):
    def test_regen_potion_applies_status(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=1, max_steps=5, render_enabled=False)
        )
        env.reset(seed=21)
        aid = "agent_0"
        env.state.agents[aid].inventory.append("regen_potion")
        _, _, _, _, info = env.step({aid: ACTION_USE})
        self.assertTrue(
            any(str(evt).startswith("status_apply:regen:") for evt in info[aid]["events"])
        )

    def test_use_can_cast_spell(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=2, max_steps=5, render_enabled=False)
        )
        env.reset(seed=22)
        env.state.agents["agent_0"].position = (4, 4)
        env.state.agents["agent_1"].position = (4, 5)
        env.state.agents["agent_0"].mana = 10
        _, _, _, _, info = env.step({"agent_0": ACTION_USE, "agent_1": ACTION_WAIT})
        self.assertTrue(any(str(evt).startswith("cast:") for evt in info["agent_0"]["events"]))

    def test_skill_book_teaching_rule(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=2, max_steps=5, render_enabled=False)
        )
        env.reset(seed=23)
        a0 = env.state.agents["agent_0"]
        a1 = env.state.agents["agent_1"]
        a0.position = (4, 4)
        a1.position = (4, 5)
        a0.faction_id = 1
        a1.faction_id = 1
        env.state.faction_leaders[1] = "agent_0"
        a0.inventory = []
        a0.skills["mining"] = 5
        token = "skill_book@test"
        env.state.item_metadata[token] = {
            "kind": "skill_book",
            "base_id": "skill_book",
            "skill_name": "mining",
            "max_teachable_level": 6,
            "uses": 2,
            "author_id": "agent_0",
            "author_level": 5,
        }
        a0.inventory.append(token)
        _, _, _, _, info = env.step({"agent_0": ACTION_GIVE, "agent_1": ACTION_WAIT})
        self.assertTrue(any(str(evt).startswith("teach:agent_1:mining:") for evt in info["agent_0"]["events"]))
        self.assertGreaterEqual(env.state.agents["agent_1"].skills.get("mining", 0), 1)

    def test_enchant_tracks_author(self):
        env = PettingZooParallelRLRLGym(
            EnvConfig(width=12, height=10, n_agents=1, max_steps=5, render_enabled=False)
        )
        env.reset(seed=24)
        aid = "agent_0"
        agent = env.state.agents[aid]
        agent.equipped.append("dagger")
        agent.inventory.append("enchant_rune")
        _, _, _, _, info = env.step({aid: ACTION_USE})
        self.assertTrue(any(str(evt).startswith("enchant:") for evt in info[aid]["events"]))
        meta = env.state.item_metadata.get("dagger", {})
        ench = list(meta.get("enchantments", []))
        self.assertTrue(any(str(row.get("by", "")) == aid for row in ench if isinstance(row, dict)))


if __name__ == "__main__":
    unittest.main()
