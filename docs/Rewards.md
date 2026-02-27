# Reward System Summary

This document summarizes the current reward shaping implemented in `rlrlgym/env.py`.

## Reward Buckets

Per-step reward details are emitted in `info[agent_id]["reward_components"]` with these buckets:

- `action_total`: direct action outcomes (move/eat/attack/use/etc.)
- `survival`: hunger/starvation pressure penalties
- `search_explore`: exploration and enemy-awareness shaping
- `profile_shape`: profile-based event shaping from `agent_profiles.json`
- `teamwork`: faction/team actions and leader share
- `treasure`: treasure held per turn and end-of-episode treasure bonus
- `focus`: objective bonuses (treasure/combat/skill/team event emphasis)
- `terminal`: death/timeout/engagement terminal terms

## Core Action Rewards

## Movement / Wait

- Valid movement gives a small base reward and move-bias reward.
- Movement pays a small step cost and potential encumbrance penalty.
- Moving toward food gives bonus; moving away gives penalty.
- Exploration of new visited tiles gives reward.
- Anti-loop penalties apply (stutter/repeat/wait loop).
- Invalid movement (bump) is penalized.

## Loot / Pickup / Eat / Use / Equip

- Loot/pickup from ground/chests/food cache grants positive reward.
- Failed loot attempts are penalized.
- Eating grants reward proportional to hunger restored.
- Wasteful eating near full hunger is penalized.
- Using medical items to heal gives positive reward.
- Equipping items gives a small positive reward.

## Combat

- Successful damage/kill gives positive reward.
- Misses/blocked/no-target attacks are penalized.
- Friendly fire is allowed but penalized:
  - penalty per HP of allied damage
  - larger penalty for killing an ally

## Team / Faction Rewards

All agents start as solo (`faction_id = -1`).

Faction interaction/actions:

- `interact` can create faction, invite adjacent agent, or join via pending invite context.
- Explicit actions support:
  - `accept_invite`
  - `leave_faction`
  - `give`
  - `trade`
  - `revive`
  - `guard`

Team reward sources:

- faction create/invite/join/leave
- give/trade/revive/guard
- team proximity bonus when adjacent to an ally

Leader share:

- each faction leader receives a small share (`leader_team_share`) of team-generated rewards created by that faction during the step.

## Treasure Objective Rewards

Treasure items are defined in `data/items.json` via `"is_treasure": true`.

- Per turn held: each living agent gets `treasure_hold_reward_per_turn` per treasure item held (inventory + equipped), up to `treasure_hold_reward_cap_items`.
- End-of-episode payout: living agents receive `treasure_end_bonus_per_item` for each treasure item still held.

## Exploration / Search / Enemy Awareness

- Reward for newly seen tiles in current observation window.
- Frontier-step bonus when moving near unseen regions.
- Stagnation penalty after prolonged no-discovery.
- Enemy awareness shaping:
  - first enemy seen bonus
  - per-step enemy visible reward
  - reward/penalty based on enemy distance delta
  - penalty for losing sight of enemy

## Survival Pressure

- Hunger decreases over time (if hunger ticks enabled).
- Low hunger pressure adds penalty.
- At zero hunger, starvation damage and penalty apply.

## Formation / Guard Effects (indirect reward impact)

These mechanics primarily affect combat outcomes:

- Formation bonus with adjacent allies:
  - small hit-chance bonus for attacker
  - small DR bonus for defender
- Guard reduces incoming damage for a protected adjacent ally.

These increase downstream rewards by reducing damage taken and improving team combat outcomes.

## Terminal Rewards / Penalties

- Death: terminal penalty.
- End-of-episode engagement bonus if combat exchanges occurred.
- If no combat exchanges occurred, an `episode_no_combat` event is recorded.
- Timeout tie penalty at max steps.
- Additional timeout-no-contact event can be emitted when no enemy was ever seen/engaged.

## Profile-Based Reward Shaping

`AgentProfile.reward_adjustment(...)` applies additional reward deltas from profile event weights (`data/agent_profiles.json`) on top of the base system.

