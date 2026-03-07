# Construction System

This document describes how construction currently works in the environment.

## Overview

Construction is implemented as a recipe variant:

- A construction recipe sets `build_tile_id` (and may leave `outputs` empty).
- Construction is triggered through station interaction via `ACTION_INTERACT`.
- Current base construction recipes are wall placements (`wood_wall`, `rock_wall`).

Core runtime logic is in:

- `MultiAgentRLRLGym._interact_station(...)`
- `MultiAgentRLRLGym._build_from_recipe(...)`

## Construction Recipe Fields

Construction uses standard recipe fields plus:

- `build_tile_id`: tile id to place in the world.

Validation guarantees:

- `build_tile_id` must reference a known tile id.
- Referenced input items must exist.

## Runtime Build Flow

When a station selects a recipe with `build_tile_id`:

1. Recipe inputs are consumed.
2. Build placement is attempted on adjacent tiles (N/S/E/W) around the builder.
3. On success:
   - target tile is replaced with `build_tile_id`
   - event emitted: `build:<tile_id>:<r>:<c>`
4. On failure:
   - inputs are refunded
   - event emitted: `build_fail:no_space`

## Placement Constraints

A build target must:

- Be inside non-border map coordinates.
- Not be occupied by:
  - living agent
  - living monster
  - chest
  - station
  - resource node

## Current Base Construction Content

Base recipes:

- `craft_wood_wall` -> places `wood_wall`
- `craft_rock_wall` -> places `rock_wall`

Both are currently unlocked through `workbench` station spawns.

## Skills and Progression

Construction recipes currently use:

- `crafting` skill (with `min_skill` gating)

Skill XP is granted when construction succeeds.

## Scenario Authoring

For scenario-driven construction content:

- Define build recipes in bundled `recipes_data`.
- Ensure `build_tile_id` entries exist in bundled `structures_data` (tiles).
- Ensure station spawns unlock the intended construction recipes.
