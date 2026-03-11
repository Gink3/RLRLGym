# Environment Performance Plan

## Goal

Reduce environment step time enough that curriculum and replay smoke tests are fast to iterate, while keeping the environment in Python unless profiling proves a compiled kernel is necessary.

## Current Problem

The environment layer in [env.py](/proj/RLRLGym/rlrlgym/world/env.py) appears to be the training bottleneck, especially as curriculum phases grow in map size and system count. Likely hotspots from code inspection:

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

## Phase 1: Measure

### Deliverables

- A repeatable profiling command using the smoke curriculum runner.
- One saved flamegraph from `py-spy` with subprocess support when relevant.
- A short hotspot summary checked into a note or attached to an issue.

### Commands

Run the smoke training job:

```bash
./scripts/train_crafting_curriculum_10_agents_smoke.sh
```

Profile a direct Python training process:

```bash
python3 -m train \
  --backend rllib \
  --algo ppo_masked \
  --iterations 7 \
  --scenario-path data/scenarios/crafting_curriculum_10_agents \
  --curriculum-path /tmp/crafting_curriculum_smoke.json \
  --shared-policy \
  --num-rollout-workers 0 \
  --train-batch-size 400 \
  --sgd-minibatch-size 128 \
  --num-sgd-iter 1 \
  --rollout-fragment-length 50 \
  --sample-timeout-s 300 \
  --replay-save-every 1 \
  --seed 0 \
  --no-aim \
  --output-dir outputs/train/crafting_curriculum_10_agents_smoke &
PY_PID=$!
sudo .venv/bin/py-spy record --pid "$PY_PID" --duration 20 -o pyspy.svg
```

### Success Criteria

- We can name the top 5 functions by sampled wall time.
- We can separate environment cost from RLlib overhead.

## Phase 2: Cheap Python Wins

### 1. Occupancy cache

Replace repeated scans over all agents, monsters, and animals inside `_walkable` and related helpers with maintained occupancy sets/maps.

Target changes:

- Maintain live occupied tile sets for agents, monsters, animals.
- Update them incrementally on movement, death, revive, spawn.
- Use O(1) membership checks in pathing and walkability helpers.

Expected payoff:

- High

Risk:

- Medium, because stale occupancy state can create movement bugs.

### 2. Visibility caching per step

Avoid recomputing LOS-heavy visibility multiple times for the same agent within one step.

Target changes:

- Cache visible tiles for each agent once per step.
- Cache enemy-visible boolean once per step.
- Reuse cached values for:
  reset metrics,
  search rewards,
  observation building,
  info payloads.

Expected payoff:

- High

Risk:

- Low to medium

### 3. Opaque-grid cache

Convert repeated tile-id lookups and opacity checks into a boolean grid or flat array that is cheap to access during LOS.

Expected payoff:

- Medium to high

Risk:

- Low

### 4. Replay snapshot reduction

Replace `copy.deepcopy(self.state)` in replay capture with a dedicated snapshot serializer that copies only replay-visible state.

Expected payoff:

- Medium, especially when `replay_save_every=1`

Risk:

- Medium

## Phase 3: Algorithmic Improvements

### 1. Precomputed LOS rays

For a given observation window, precompute ray offsets from the center and reuse them across agents rather than rebuilding line point lists repeatedly.

### 2. Cheaper nearest-opponent queries

Current opponent distance checks are repeated often. Evaluate:

- per-step pairwise distance cache for agents
- cheap spatial bucketing by coarse grid cell

### 3. Observation-path split

Separate simulation logic from observation construction so simulation steps can be profiled independently of RL observation serialization.

## Phase 4: Structural Python Refactors

### 1. Reduce object churn

- Reuse per-step buffers where practical
- Minimize transient `dict`, `list`, and `set` creation in hot loops
- Avoid converting between tuples, lists, and dicts repeatedly in step-critical code

### 2. Flatten hot entity data

For the hottest simulation paths, consider moving from nested dataclass access toward indexed arrays or compact state records for:

- positions
- hp
- alive flags
- faction ids

This should be done only for hot-path data, not the entire codebase.

## Phase 5: Compiled Python Before Rust

If Python-side algorithmic work is not enough, try a compiled boundary before a full rewrite.

### Preferred order

1. Cython for LOS, walkability, visibility, and combat helpers
2. mypyc for type-stable pure-Python modules
3. Numba only if hot paths are converted to numeric array-oriented code

### Constraint

Only move isolated kernels first. Do not port content loading, replay serialization, RLlib adapters, or scenario configuration prematurely.

## Phase 6: Rust Only If Needed

Rust becomes justified if all of the following are true:

- Profiling still shows the simulation kernel dominates runtime after Phases 2 through 5.
- Hot paths are well isolated and stable.
- Python orchestration can call into a Rust step kernel through a narrow API.
- The team is willing to absorb the build, packaging, and debugging overhead.

Preferred Rust scope:

- simulation step kernel only
- occupancy and LOS
- combat resolution

Keep in Python:

- config and JSON loading
- scenarios and curriculum wiring
- replay viewers and tools
- training scripts and RLlib integration

## Immediate Next Tasks

1. Generate a `py-spy` flamegraph from the 7-phase smoke run.
2. Implement occupancy caches.
3. Implement per-step visibility caching.
4. Re-profile and compare against the baseline.
5. Only then decide whether compiled helpers are necessary.

## Done Criteria

- Smoke curriculum runtime is materially shorter.
- Replay-enabled smoke runs remain practical.
- Phase transitions no longer feel blocked by environment step time.
- We have before/after profiles showing where the win came from.
