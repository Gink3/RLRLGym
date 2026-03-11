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

Success criteria:

- We can name the top 5 functions by sampled wall time.
- We can separate environment cost from RLlib overhead.

### Version 0.2

Focus: cheap Python wins in movement and collision checks.

Features / tasks:

- Add occupancy caches for agents, monsters, and animals.
- Replace repeated scans inside `_walkable` and related helpers with O(1) membership checks.
- Update occupancy state incrementally on movement, spawn, death, and revive.

Success criteria:

- Movement and collision checks no longer scan all live entities on every query.
- Profiling shows a measurable drop in wall time for walkability-related helpers.
- Behavior remains correct for movement, spawn, death, and revive paths.

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

Success criteria:

- Visibility and enemy-visible values are computed once per agent per step.
- Profiling shows a measurable reduction in LOS and visibility hotspot cost.
- Observation and reward logic still agree on what an agent can see.

### Version 0.4

Focus: algorithmic improvements in repeated geometric and search work.

Features / tasks:

- Precompute LOS ray offsets for common observation windows.
- Reduce repeated nearest-opponent calculations.
- Evaluate per-step pairwise distance caches or spatial bucketing.
- Split simulation work from observation construction for clearer profiling.

Success criteria:

- Repeated geometric work is reduced in the main step loop.
- Nearest-opponent and LOS-heavy paths are cheaper in profiling than the previous version.
- Simulation work can be measured separately from observation construction.

### Version 0.5

Focus: replay and allocation reduction.

Features / tasks:

- Replace `copy.deepcopy(self.state)` replay capture with a replay-specific serializer.
- Reduce transient `dict`, `list`, and `set` allocations in hot loops.
- Reuse per-step buffers where practical.

Success criteria:

- Replay-enabled smoke runs are materially cheaper than the baseline.
- Allocation-heavy hot loops show reduced sampled time or reduced call frequency.
- Replay output remains correct and usable in the viewer.

### Version 0.6

Focus: structural Python refactors.

Features / tasks:

- Flatten hot-path entity data into cheaper internal representations where justified.
- Reduce nested dataclass access in the hottest simulation loops.
- Keep external APIs stable while simplifying the sim kernel internally.

Success criteria:

- Hot-path entity access is simpler and cheaper in profiling.
- Public environment behavior and external APIs remain unchanged.
- The sim kernel is easier to isolate for future compiled backends.

### Version 0.7

Focus: compiled Python before Rust.

Features / tasks:

1. Try Cython for LOS, walkability, visibility, and combat helpers.
2. Evaluate mypyc for type-stable pure-Python modules.
3. Use Numba only if hot paths are converted to numeric array-oriented code.

Success criteria:

- At least one compiled-Python option is tested against a real hotspot.
- Only isolated kernels are moved; orchestration and content loading remain in Python.
- The compiled path provides enough improvement to justify its maintenance cost.

### Version 0.8

Focus: Rust only if Python and compiled-Python options are still insufficient.

Features / tasks:

- Define the minimum Rust scope if Python-side work is still insufficient.
- Keep any future Rust boundary limited to the simulation step kernel and other proven hotspots.
- Keep config, scenarios, replay tooling, and RLlib integration in Python.

Success criteria:

- Rust is only pursued if profiling still shows the simulation kernel dominates after versions `0.2` through `0.7`.
- Hot paths are isolated enough to support a narrow Python-to-Rust boundary.
- The expected speedup is large enough to justify build and maintenance overhead.

## Immediate Next Tasks

1. Fill out feature bullets under each version as implementation decisions get locked in.
2. Generate a `py-spy` flamegraph from the 7-phase smoke run.
3. Start version `0.2` with occupancy caches.
4. Re-profile after each version increment.

## Done Criteria

- Smoke curriculum runtime is materially shorter.
- Replay-enabled smoke runs remain practical.
- Phase transitions no longer feel blocked by environment step time.
- We have before/after profiles showing where the win came from.
