# Development Plan

## Goal

Reduce environment step time enough that curriculum and replay smoke tests are fast to iterate, while keeping the environment in Python unless profiling proves a compiled kernel is necessary.

## Current Problem

The environment layer in [env.py](/proj/RLRLGym/rlrlgym/world/env.py) appears to be the training bottleneck, especially as curriculum phases grow in map size and system count.

Likely hotspots from code inspection:

- Line-of-sight and visibility:
  [env.py](/proj/RLRLGym/rlrlgym/world/env.py)
  `_visible_tile_coords`, `_enemy_visible`, `_has_line_of_sight`
- Occupancy checks:
  [env.py](/proj/RLRLGym/rlrlgym/world/env.py)
  `_walkable`, `_walkable_for_monster`
- Observation building and repeated per-agent metric recomputation:
  [env.py](/proj/RLRLGym/rlrlgym/world/env.py)
  `_build_observation`, search/exploration reward helpers
- Replay capture:
  [env.py](/proj/RLRLGym/rlrlgym/world/env.py)
  `capture_playback_state` via `copy.deepcopy`

## Guiding Principles

- Optimize based on measured hotspots, not guesses.
- Prefer algorithmic and data-layout wins before changing language.
- Keep Python orchestration, config loading, RLlib integration, and replay tooling intact.
- Isolate hot kernels so they can be compiled later without rewriting the whole environment.

## Versioned Roadmap

### Version 0.1

Focus: measurement baseline.

Features / tasks:

- Add a repeatable profiling command using the smoke curriculum runner.
- Capture at least one `py-spy` flamegraph for the environment-heavy smoke run.
- Write down the top 5 sampled hotspots before making optimizations.

### Version 0.2

Focus: cheap Python wins in movement and collision checks.

Features / tasks:

- Add occupancy caches for agents, monsters, and animals.
- Replace repeated scans inside `_walkable` and related helpers with O(1) membership checks.
- Update occupancy state incrementally on movement, spawn, death, and revive.

### Version 0.3

Focus: visibility and line-of-sight caching.

Features / tasks:

- Cache visible tiles per agent once per step.
- Cache enemy-visible booleans per agent once per step.
- Reuse cached visibility in:
  reset metrics,
  search rewards,
  observation building,
  info payloads.
- Add an opaque-grid cache for fast LOS checks.

### Version 0.4

Focus: algorithmic improvements in repeated geometric and search work.

Features / tasks:

- Precompute LOS ray offsets for common observation windows.
- Reduce repeated nearest-opponent calculations.
- Evaluate per-step pairwise distance caches or spatial bucketing.
- Split simulation work from observation construction for clearer profiling.

### Version 0.5

Focus: replay and allocation reduction.

Features / tasks:

- Replace `copy.deepcopy(self.state)` replay capture with a replay-specific serializer.
- Reduce transient `dict`, `list`, and `set` allocations in hot loops.
- Reuse per-step buffers where practical.

### Version 0.6

Focus: structural Python refactors.

Features / tasks:

- Flatten hot-path entity data into cheaper internal representations where justified.
- Reduce nested dataclass access in the hottest simulation loops.
- Keep external APIs stable while simplifying the sim kernel internally.

### Version 0.7

Focus: compiled Python before Rust.

Features / tasks:

1. Try Cython for LOS, walkability, visibility, and combat helpers.
2. Evaluate mypyc for type-stable pure-Python modules.
3. Use Numba only if hot paths are converted to numeric array-oriented code.

### Version 0.8

Focus: Rust only if Python and compiled-Python options are still insufficient.

Features / tasks:

- Define the minimum Rust scope if Python-side work is still insufficient.
- Keep any future Rust boundary limited to the simulation step kernel and other proven hotspots.
- Keep config, scenarios, replay tooling, and RLlib integration in Python.

## Immediate Next Tasks

1. Fill out feature bullets under each version as implementation decisions get locked in.
2. Generate a `py-spy` flamegraph from the 7-phase smoke run.
3. Start version `0.2` with occupancy caches.
4. Re-profile after each version increment.
