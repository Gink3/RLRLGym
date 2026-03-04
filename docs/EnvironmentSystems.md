# Environment Skills, Systems, Observation, and Reward Spaces

This document summarizes the major systems currently implemented in `rlrlgym/env.py` and related data files.

## Agent Skill System

Agents track level + XP for these skills:

- `melee`
- `archery`
- `thrown_weapons`
- `medic`
- `athletics`
- `exploration`
- `mining`
- `woodcutting`
- `crafting`
- `smithing`
- `alchemy`
- `farming`
- `armor_light`
- `armor_medium`
- `armor_heavy`

Skill levels drive core mechanics:

- Combat hit/damage/defense and armor-class mitigation XP
- Movement/encumbrance mitigation (`athletics`)
- Fog/search effectiveness (`exploration`)
- Gathering yield (`mining`, `woodcutting`)
- Planting efficiency and seed economy (`farming`)
- Crafting/enchant/writing outcomes (`crafting`, `smithing`, `alchemy`)

## Major Gameplay Systems

### World and Content

- Procedural terrain with biomes and tile-weighted generation
- Resource nodes with skill-gated gathering
- Station spawning with recipe unlocks/bonuses
- Monsters with combat + loot tables
- Animals (cow, sheep, pig, chicken) with ecology loops

### Economy and Inventory

- Item definitions are data-driven
- Weight-based carrying/dragging and over-encumbrance penalties
- Ground items, chest loot, monster drops, animal material drops
- Coin/material/tool/potion/book/rune style item support

### Crafting, Writing, and Enchanting

- Recipe system (inputs -> outputs, station requirements, optional build tile)
- Skill books with metadata (`skill_name`, `max_teachable_level`, `uses`, `author_id`, `author_level`)
- Teaching rule capped by author and book constraints
- Enchantments are data-driven and stack-limited; each applied enchant tracks who applied it
- Crafting details: [CraftingSystem.md](/proj/RLRLGym/docs/CraftingSystem.md)
- Construction details: [ConstructionSystem.md](/proj/RLRLGym/docs/ConstructionSystem.md)

### Combat, Status, and Spells

- Melee/ranged style attack flow with accuracy/evasion/DR
- Component-style status effects (`apply`, tick cadence, expiry behavior)
- Spells use shared effect handling with costs (mana/cooldown/reagents)
- Utility/combat effects include direct damage/heal, status apply/clear, movement/cleanse-style effects
- Combat details and spell catalog: [Combat.md](/proj/RLRLGym/docs/Combat.md)

### Faction / Team System

- Faction creation, invite, accept, leave, guard, revive, give, trade
- Team-based reward sharing and leadership bonuses
- Anti-loop protections for repeated team reward farming

### Animal Ecology

- Animals tick hunger/thirst every turn
- Animals forage from environmental tiles (`grass`, `bush`, `tree`) and drink near water
- Forage is consumed/depleted from the environment over time
- Mature animals can reproduce when needs are met
- Starvation/dehydration kills animals
- Sheep can be sheared and regrow wool over time

## Per-Agent Observation Space

`env.observation_space(agent_id)` returns a dict descriptor. Runtime observations are per-agent dictionaries.

Always present keys:

- `step`
- `alive`
- `profile`
- `race`
- `class`
- `faction`:
  - `faction_id`
  - `is_leader`
  - `pending_invite_from_faction`

Optional keys by profile/config:

- `local_tiles`: local tile window based on dynamic view size
- `stats`: rich state/features including:
  - Core vitals/resources: `hp`, `mana`, `max_mana`, `hunger`
  - Position/loadout: `position`, `equipped_count`, `armor_slots`
  - Carry model: `carried_weight`, `carry_capacity`, `encumbrance_ratio`
  - Attributes/skill state: `strength`, `dexterity`, `intellect`, `skills`, `skill_xp`, `overall_level`
  - Spell/status state: `known_spells`, `spell_cooldowns`, `statuses`
  - Team/visibility/search metrics and nearby entity summaries
  - Nearby summaries include chests, resource nodes, stations, agents, monsters, animals
- `inventory`: current carried item IDs

## Per-Agent Reward Space

Each step returns one scalar reward per agent:

- `rewards[agent_id]` is a float total reward

The reward is compositional. Per-step bucket breakdown is exposed in:

- `info[agent_id]["reward_components"]`

Current buckets:

- `action_total`
- `survival`
- `search_explore`
- `profile_shape`
- `teamwork`
- `treasure`
- `focus`
- `terminal`

Reward includes:

- Direct action outcomes (move, interact, gather, craft, attack, social actions)
- Survival pressure (hunger, starvation, status ticking)
- Exploration/search shaping and enemy visibility/distance shaping
- Team/faction effects and leader-share mechanics
- Treasure hold/end-of-episode bonuses
- Terminal outcomes (death, timeout-tie, engagement bonuses)
- Profile-specific shaping adjustments via `AgentProfile.reward_adjustment(...)`

For full bucket-level details, see [Rewards.md](/proj/RLRLGym/docs/Rewards.md).
