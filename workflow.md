# Scenario Content Workflow

## Goal
Build self-contained scenarios where item, monster, animal, structure, and probability-bearing definitions are bundled into the scenario itself so training can run from a single selected scenario directory.

## Authoring Order
1. Define core entities.
2. Define spawn tables / probability weights.
3. Build a scenario that bundles everything.
4. Run training from the bundled scenario.

## 1) Define Core Entities
Use Scenario Editor (`python3 tools/scenario_editor.py`) and edit the `Scenario env_config` JSON field.

Update these embedded sections:
- `items_data.items`: all item definitions.
- `monsters_data.monsters`: all monster definitions.
- `structures_data.tiles`: all structure/tile definitions (including `spawn_weight` and tile `loot_table`).
- `recipes_data.recipes`: all crafting and construction recipes.
- `statuses_data.statuses`: all status-effect definitions.
- `spells_data.spells`: all spell definitions.
- `enchantments_data.enchantments`: all enchant definitions and stack rules.
- `animals_data.animals`: livestock/wildlife definitions and ecology traits (`drop_item`, maturity, reproduction, shearing).

Notes:
- Keep `schema_version` in each embedded section.
- Keep IDs consistent (`item.id`, `monster.id`, `tile.id`) because spawn tables reference them.

## 2) Define Spawn Tables / Weights
Update the probability-bearing sections in the same JSON:
- `monster_spawns_data.spawns`: each row maps `monster_id` -> `weight`.
- `mapgen_config_data.mapgen`: density controls (`monster_density`, `chest_density`) and mapgen parameters.
- `mapgen_config_data.mapgen.animal_density`: animal population density.
- `mapgen_config_data.mapgen.biomes`: biome weights and biome-local terrain tile weights.
- `mapgen_config_data.mapgen.resource_nodes`: gathering node definitions (skill, drop item, density, yield).
- `mapgen_config_data.mapgen.station_spawns`: station placement and unlocked recipe sets.
- `items_data.chest_loot_table`: chest loot sampling pool.
- `structures_data.tiles[*].spawn_weight`: relative map spawn probability for each structure/tile.
- `monsters_data.monsters[*].loot[*].weight`: per-monster loot probability weights.

## 3) Bundle Into Scenario
Save the scenario in Scenario Editor.

A saved scenario directory contains:
- `env_config.json`
- `agents.json`

`env_config.json` now includes embedded bundled payloads:
- `structures_data`
- `items_data`
- `monsters_data`
- `monster_spawns_data`
- `mapgen_config_data`
- `recipes_data`
- `statuses_data`
- `spells_data`
- `enchantments_data`
- `animals_data`

These bundled sections are used at runtime and do not require external base-data files.

## 4) Train Using the Scenario
Open Training Launcher (`python3 tools/train_launcher.py`), select the scenario directory, then start training.

Equivalent CLI:
```bash
python3 -m train --scenario-path data/scenarios/<your_scenario_dir> --backend custom
```

## Validation Checklist
Before training, verify:
- Every `monster_spawns_data.spawns[*].monster_id` exists in `monsters_data.monsters[*].id`.
- Every loot item in monster and structure tables exists in `items_data.items[*].id`.
- Every `resource_nodes[*].drop_item` and every recipe input/output item exists in `items_data.items[*].id`.
- Every recipe `build_tile_id` exists in `structures_data.tiles[*].id`.
- Every `animals_data.animals[*].drop_item` and `shear_item` exists in `items_data.items[*].id`.
- `schema_version` exists in each embedded section.
- At least one monster spawn entry exists and has positive weights.

## Compatibility Notes
- Existing path keys (`*_path`) may still be present, but bundled `*_data` sections are preferred by the runtime.
- `structures_data` is treated as the tile/structure source for map generation.
