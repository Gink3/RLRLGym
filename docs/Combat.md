# Combat System

This document describes combat resolution, status effects, and spells in the current environment.

## Combat Overview

Combat happens through:

- `ACTION_ATTACK` for agent-initiated attacks
- Monster turns (`_apply_monster_turn`) for monster attacks/movement
- Status/spell effects that apply direct damage, healing, control, or utility

Attack priority for `ACTION_ATTACK`:

1. Adjacent monster
2. Adjacent animal
3. Adjacent agent
4. If no valid adjacent target: `attack_no_target`

## Agent Attack Resolution

Core path: `_attack_agent(...)`

1. Weapon selection:
   - Uses most recently equipped weapon if present
   - Falls back to unarmed (`blunt`, `1..2`, `melee`)
2. Hit chance:
   - Base from `_hit_chance(...)` with attacker dexterity/skill and defender dexterity
   - Modified by formation/team adjacency bonuses
3. Raw damage:
   - Weapon range roll + stat bonus + skill scaling
4. Defender DR:
   - `_roll_hit_location_dr(...)` chooses hit slot and sums:
     - race base DR and per-damage-type DR
     - armor DR on hit slot
     - enchant defense bonus
     - armor skill bonus by armor class (`armor_light|medium|heavy`)
     - defend stance and formation DR
5. Final damage:
   - `max(0, raw - dr)` plus attacker enchant damage bonus
   - May be reduced by guard interception (`team_guard_block`)
6. Outcomes:
   - Hit applies HP loss, events, and rewards/penalties
   - Friendly fire applies ally penalties
   - Kills mark target dead and apply kill rewards/penalties
   - On some non-lethal hostile hits, bleed may be applied

## Monster Attack Resolution

Core path: `_monster_attack(...)`

- Hit chance scales with monster `acc` and target dex/athletics
- Damage is blunt by default
- Uses same defender DR/hit-location system as agent-vs-agent
- Can apply poison on successful non-lethal hit (chance-based)
- Lethal hits terminate agent and emit death-by-monster events

## Armor, Defense, and Team Mitigation

Defense layers:

- Race base DR + race damage-type modifiers
- Equipped armor DR by hit location
- Armor class skill mitigation:
  - `armor_light`
  - `armor_medium`
  - `armor_heavy`
- `ACTION_DEFEND` temporary DR bonus
- Guard ally interception reduction
- Formation DR bonus with adjacent allies

## Status Effect System

Statuses are component/data-driven and tracked as active components:

- `duration`
- `tick_interval`
- `apply` effects
- `tick` effects
- `expire` effects
- tags (for behavior gates like `confuse` and `paralyze`)

Runtime behavior:

- `_apply_status(...)` applies or refreshes a status
- `_tick_agent_statuses(...)` decrements and executes tick/expire effects
- `_clear_status(...)` removes a status and runs expire/cleanse behavior
- `_status_adjust_action(...)` enforces behavior constraints:
  - `paralyze` forces `WAIT`
  - `confuse` can scramble action

Status resistance:

- `resistance` status tag grants resist chance
- enchantment `status_resist` contributes additional resist chance

Current base statuses (`data/base/statuses.json`):

- `poison` (DoT)
- `bleed` (DoT)
- `regen` (healing over time)
- `confused` (action scramble)
- `paralyzed` (action lock to wait)
- `resistance` (status resistance buff)
- `haste` (mana gain over time)

## Spell System

Spells are data-driven and executed via `_cast_best_spell(...)`.

Cast gating:

- Spell is known by caster
- Enough mana
- Cooldown is zero
- Required reagents present (if any)
- Valid target in range for spell target mode (`self`, `ally`, `enemy`)

On cast:

- Reagents are consumed
- Mana cost applied
- Cooldown set
- Spell effects executed via shared `_apply_effects(...)`
- Cast event emitted and caster gains alchemy XP

### Supported Effect Types

- `damage`
- `heal`
- `hunger`
- `mana`
- `apply_status`
- `cure_status`
- `cleanse`
- `reveal`
- `teleport_blink`
- `knockback`

### Current Base Spells (`data/base/spells.json`)

- `arc_bolt`: enemy ranged direct damage
- `cleanse`: self cleanse (poison/bleed/confused/paralyzed)
- `blink`: self short teleport to nearby walkable tile
- `haste`: self applies haste status
- `regen_touch`: ally applies regen status
- `reveal`: self utility reveal event

### Default Known Spells by Class

- `fighter`: `arc_bolt`, `cleanse`, `haste`
- `rogue`: `arc_bolt`, `blink`, `reveal`
- `medic`: `cleanse`, `regen_touch`, `reveal`

## Combat Consumables

Combat-relevant items in `ACTION_USE` include:

- `bandage`: healing
- `healing_potion`: stronger healing
- `antidote`: clears poison
- `cleanse_potion`: clears poison/bleed/confused/paralyzed
- `regen_potion`: applies regen status
- `resistance_tonic`: applies resistance status

## Observability and Debugging

Combat emits detailed event logs in `info[agent_id]["events"]`, including:

- attack/miss/hit/kill rolls
- DR and hit slot details
- status apply/expire/cleanse
- spell cast and effect traces
- guard/formation mitigation events

Reward bucket breakdown is in `info[agent_id]["reward_components"]`.
