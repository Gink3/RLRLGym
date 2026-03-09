# Crafting System

This document describes how crafting currently works in the environment.

## Overview

Crafting is station-driven and recipe-driven:

- Recipes are loaded from `data/base/recipes.json` (or bundled `recipes_data` in a scenario).
- Stations are spawned from mapgen (`station_spawns`) and expose recipe subsets.
- Agents craft by using `ACTION_INTERACT` while standing on a station tile.

Core runtime logic is in:

- `MultiAgentRLRLGym._interact_station(...)`
- `MultiAgentRLRLGym._best_station_recipe(...)`

## Recipe Data Model

Each recipe supports:

- `id`: unique recipe id
- `inputs`: required item counts
- `outputs`: produced item counts
- `skill`: gating/progression skill (`crafting`, `smithing`, etc.)
- `min_skill`: required minimum level
- `station`: station id (for example `workbench`, `smelter`, `mint`)
- `craft_time`: currently metadata only (not an asynchronous build timer yet)
- `speed_multiplier`: included in craft event output; used for current speed signal
- `quality_bonus`: metadata field (not directly applied yet)
- `tags`: category labels

Validation guarantees:

- Inputs must be non-empty.
- At least one of `outputs` or `build_tile_id` must be provided.
- Referenced item ids are verified at env startup.

## Station Data Model

Stations come from mapgen `station_spawns`:

- `id`
- `density`
- `speed_multiplier`
- `quality_tier`
- `unlock_recipes`

At runtime:

- `unlock_recipes` restricts which recipes can be selected at that station.
- `quality_tier` adds bonus output quantity for item-output recipes.
- Effective craft speed shown in events is `station.speed_multiplier * recipe.speed_multiplier`.

## Craft Selection and Execution Flow

When an agent uses `ACTION_INTERACT` on a station:

1. Candidate recipes are filtered by:
   - Station id match
   - Inclusion in station `unlock_recipes` (if non-empty)
   - Agent skill level >= `min_skill`
   - Agent inventory has all inputs
2. Highest-priority candidate is selected (`min_skill` then id ordering).
3. Inputs are consumed.
4. Output items are created, with station `quality_tier` bonus.
5. Skill XP is awarded for the recipe skill.
6. Craft event is emitted, including station id and effective speed.

If no recipe is craftable:

- Event: `station_idle:<station_id>`
- Small negative reward is applied.

## Current Crafting Recipe Set (Base Data)

Base crafting recipes include:

- Smelting: copper/iron/silver/gold ingots at `smelter`
- Coinage: copper/silver/gold coins at `mint` (from ingots)

## Skills and Progression

Crafting progression currently uses:

- `crafting` (general craft)
- `smithing` (smelting/minting recipes)

Related systems:

- Gathering (`mining`, `woodcutting`) supplies recipe inputs.
- Carry capacity/encumbrance affects hauling and logistics.
- Enchanting/books integrate with crafted and equipped items.

## Scenario Authoring

For scenario-driven content:

- Bundle `recipes_data` directly in scenario `env_config.json`.
- Bundle corresponding `mapgen_config_data.mapgen.station_spawns`.
- Ensure referenced items exist in bundled `items_data`.
