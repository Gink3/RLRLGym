# Animal Ecology System

This document describes the current animal ecology loop in `RLRLGym`.

## Overview

Animal ecology is a per-step simulation that runs after agent and monster turns. Each alive animal:

1. Ages and updates cooldown/regrow timers.
2. Moves (`movement_speed` steps per turn), prioritizing hunt/food/water.
3. Ticks hunger/thirst and consumes nearby resources when possible.
4. Attempts reproduction when maturity and needs thresholds are met.

Core runtime entry points:

- `MultiAgentRLRLGym._apply_animal_turn(...)`
- `MultiAgentRLRLGym._spawn_animals(...)`
- `MultiAgentRLRLGym._animal_move(...)`
- `MultiAgentRLRLGym._animal_hunt_step(...)`
- `MultiAgentRLRLGym._animal_try_reproduce(...)`

## Content Source

Animal definitions live in:

- `data/base/animals.json`

Key per-animal fields:

- Identity/render: `id`, `name`, `symbol`, `color`
- Spawn: `spawn_weight`
- Survival/combat: `hp`, `max_hunger`, `max_thirst`, `movement_speed`, `prey_score`, `carnivore`
- Reproduction: `litter_size_min`, `litter_size_max`, `mature_age`, `reproduction_cooldown`
- Harvesting: `drop_item`, `can_shear`, `shear_item`, `shear_regrow_steps`

## Spawning

Animals spawn during `reset()` using mapgen `animal_density` and walkable non-occupied tiles:

- Candidate species are chosen with weighted random by `spawn_weight`.
- Initial state starts at roughly half hunger/thirst.
- Gender is randomly assigned (`female`/`male`).
- Reproduction cooldown is randomized from `0..reproduction_cooldown`.

## Needs and Resource Pressure

Each animal turn:

- `hunger -= 1`
- `thirst -= 1`

Then:

- Herbivores attempt to consume forage in local neighborhood (`self + N/S/E/W`) using `animal_forage_tile_ids`.
- Animals near/in water restore thirst.
- If hunger or thirst reaches `0`, the animal dies and drops material.

Forage consumption depletes tiles via `tile_interactions`; when exhausted, tile converts to mapgen floor fallback.

## Movement Behavior

Animals move only to valid walkable tiles not occupied by other animals.

Priority:

1. Carnivore hunt step (if target available).
2. If thirsty, bias toward positions that can drink.
3. If hungry, bias toward positions where food is available.
4. Otherwise random valid move.

`movement_speed` controls how many movement attempts happen each turn.

## Predator/Prey Dynamics

Carnivores search for prey with these constraints:

- Not same entity/species.
- Predator must satisfy prey score gate:
  - `predator.prey_score >= prey.prey_score + prey_score_hunt_margin`
- Within hunt range (`<= max(4, monster_sight_range)` Manhattan).
- Must have line of sight.

If adjacent, predator attacks immediately; otherwise it moves to reduce distance and may attack after moving.

On prey kill:

- Prey dies.
- Predator regains hunger.
- Prey material drop occurs through normal animal drop path.

## Reproduction

An animal can reproduce when all are true:

- Alive and mature (`age >= mature_age`)
- `reproduction_cooldown == 0`
- Hunger and thirst at/above half-threshold style checks
- Opposite-sex, same-species partner adjacent and also mature/ready

Litter size is sampled from `litter_size_min..litter_size_max`.
Offspring spawn in nearby valid cells (including diagonals), inheriting species template stats.
Both parents get reproduction cooldown reset.

## Shearing

Shear interaction supports renewable animal resources:

- Agent must share tile with shearable mature animal.
- If already sheared, regrow timer must be complete.
- On success, `shear_item` is awarded and regrow timer starts.

During animal turns, regrow timer ticks down; when it reaches zero, sheep become shearable again.

## Agent Interaction With Animals

Agents can:

- Attack animals through combat interactions (`ACTION_ATTACK` path).
- Shear eligible animals (`ACTION_INTERACT_SHEAR` or generic `ACTION_INTERACT` routing).
- Observe animal summaries in observations (`nearby_animals` block).

## Important Tuning Knobs

Main ecology controls:

- `mapgen.animal_density`
- `resource_rules.animal_forage_tile_ids`
- `resource_rules.prey_score_hunt_margin`
- Species `spawn_weight`, `movement_speed`, `prey_score`
- Species hunger/thirst/reproduction parameters

If ecology feels too explosive or too sparse, start by adjusting:

1. `animal_density`
2. Carnivore `prey_score` and `prey_score_hunt_margin`
3. Reproduction cooldown + litter sizes
4. Available forage/water tile distribution in mapgen/biomes
