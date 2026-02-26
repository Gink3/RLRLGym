# RLRL Gym


A gym to simulate brogue/nethack/Cataclysm DDA.


# High Level Design
* The game model should be expandable in order to add new tiles, items, and monsters
* The game model needs a gym-compatible api
    * Primary multi-agent API should be PettingZoo Parallel API (easiest for multi-agent training workflows)
    * The game model should handle asymmetric multi agent training
    * Optional adapters can be added later for Gymnasium-compatible single-agent wrappers
* Observations should be configurable per agent to make asymmetric experiences.
* There should be a way to view/render training or playback.
    * The renderings need to be in a new window, there will be no cli viewing capability
    * The rendering needs to support tile sets eventually, but colored ascii tiles for now is fine
    * The renderings need to have pause/play/fast forward capability
        * fast forward should support 2x and 5x speeds
    * The renderings need to be able to zoom in and out to focus on single agents
        * Zoom range should be 0 to 10 (0 = full map, higher values = tighter local focus)
    * The renderings should be optional
    * The renderings need to display tile colors in the new window

## Generic Actions
* Move
* Wait/Rest
* Loot
* Eat
* Pick up items
* Equip Items
* Use Items
* Interact with the environment

## Agent Profiles
The environment should support profile-based agents with different observation spaces and reward functions.

### Initial Profiles
* `human`
    * Broader local awareness (larger view radius)
    * Balanced rewards for exploration, loot, and interaction
* `orc`
    * Narrower local awareness (smaller view radius)
    * Higher reward emphasis on interactions, stronger penalties for idle behavior

### Profile Requirements
* Profiles must be loaded from JSON config files (not hardcoded in source).
* Profile JSON should include a `schema_version` key for compatibility and migrations.
* Each agent is assigned a profile by agent id.
* Observation shape/content can differ by profile.
* Reward shaping can differ by profile while core game rules remain shared.


## Rewards
Reward Design needs to reward interactions with the environment and other agents. To limit exploitation, the environment needs to limit how many times something can be interacted with. Carefully shaped auxiliary rewards (exploration, skill gain, resource management) to avoid exploit loops. Add penalties for degenerate behavior (stutter-step farming, infinite wait loops).


## Procedural Generation
The map generation engine should be configurable to allow for different generation probabilities. All tiles should be defined in JSON, and a schema for tiles should be defined. All maps are assumed to be on exactly 1 z level. There are no vertical elements planned.

### Tile JSON Schema
* `schema_version` (required, integer): Tile schema version for compatibility and migrations.
* `tiles` (required, array): List of tile definitions.
* Tile required fields:
    * `id` (required, string): Unique tile identifier.
    * `glyph` (required, string): Single-character glyph used for ASCII rendering.
    * `color` (required, string): Color token used by renderer.
    * `walkable` (required, boolean): Whether an agent can enter the tile.
    * `spawn_weight` (required, number): Relative probability used in map generation.
    * `max_interactions` (required, integer): Number of valid interactions before depletion.
    * `loot_table` (required, array of strings): Items available from loot interactions (can be empty).


## Performance + scaling
Vectorized environments (many parallel instances).
Profile hotspots; move heavy simulation logic to optimized code if needed.
Save/load snapshots for fast evaluation and debugging.

## Testing + reproducibility
Unit tests for core mechanics.
Property tests for invariants (no illegal state transitions).
Fixed-seed regression tests for training/eval consistency.

## Baselines + training stack
Start with PPO/A2C on simplified rules.
Add recurrent policies (LSTM/Transformer) for partial observability.
Log rich metrics (survival time, dungeon depth, cause of death, resource efficiency).

## Training Metrics
Training should stream aggregate metrics to Aim for interactive analysis.

### Metrics
* Episode return curves (team return and per-agent return)
* Win rate
* Mean survival time
* Cause-of-death histogram

### Outputs
* Aim experiment traces (episode/iteration metrics)
* Model checkpoints and replay artifacts

## Documentation
All interfaces should be documented for easier usage in the `README.md` file
