# Construction System

This document describes how construction currently works in the environment.

## Overview

Construction is implemented as a recipe variant:

- A construction recipe sets `build_tile_id` or `build_station_id` (and may leave `outputs` empty).
- Construction is triggered by local interaction (`ACTION_INTERACT`) and no longer requires standing on a station.
- Construction recipes are loaded from `data/base/construction_recipes.json`.

Core runtime logic is in:

- `MultiAgentRLRLGym._interact_construction(...)`
- `MultiAgentRLRLGym._build_from_recipe(...)`

## Construction Recipe Fields

Construction uses standard recipe fields plus:

- `build_tile_id`: tile id to place in the world.
- `build_station_id`: station id to place on the current tile.

Validation guarantees:

- `build_tile_id` must reference a known tile id.
- Referenced input items must exist.

## Runtime Build Flow

When construction selects a build recipe:

1. Recipe inputs are consumed.
2. Build placement is attempted on the builder's current tile.
3. On success:
   - target tile is replaced with `build_tile_id`, or station state is created for `build_station_id`
   - event emitted: `build:<tile_id>:<r>:<c>`
4. On failure:
   - inputs are refunded
   - event emitted: `build_fail:no_space`

## Placement Constraints

A build target must:

- Be inside non-border map coordinates.
- Not be occupied by:
  - living agent (other than the builder)
  - living monster
  - chest
  - resource node
- Existing station blocks tile-building recipes.

## Current Base Construction Content

Base recipes include:

- `build_workbench` -> creates `workbench` station on the tile
- `build_wood_wall`, `build_rock_wall`
- `build_wood_door`, `build_spike_trap`
- `build_campfire`, `build_firepit`, `build_fireplace`, `build_clay_forge`

Workstations are not auto-spawned in base mapgen; agents construct them.

## Skills and Progression

Construction recipes currently use:

- `crafting` skill (with `min_skill` gating)

Skill XP is granted when construction succeeds.

## Scenario Authoring

For scenario-driven construction content:

- Define build recipes in bundled `construction_recipes_data` (or `construction_recipes_path`).
- Ensure `build_tile_id` entries exist in bundled `structures_data` (tiles).
- Ensure `build_station_id` entries map to known station definitions in `mapgen.station_spawns`.
