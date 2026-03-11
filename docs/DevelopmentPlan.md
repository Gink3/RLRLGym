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

Commands:

```bash
./scripts/train_crafting_curriculum_10_agents_smoke.sh
```

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

Success criteria:

- We can name the top 5 functions by sampled wall time.
- We can separate environment cost from RLlib overhead.

### Version 0.2

Focus: cheap Python wins in movement and collision checks.

Features / tasks:

- Add occupancy caches for agents, monsters, and animals.
- Replace repeated scans inside `_walkable` and related helpers with O(1) membership checks.
- Update occupancy state incrementally on movement, spawn, death, and revive.

Expected payoff:

- High

Risk:

- Medium, because stale occupancy state can create movement bugs.

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

Expected payoff:

- High

Risk:

- Low to medium

### Version 0.4

Focus: algorithmic improvements in repeated geometric and search work.

Features / tasks:

- Precompute LOS ray offsets for common observation windows.
- Reduce repeated nearest-opponent calculations.
- Evaluate per-step pairwise distance caches or spatial bucketing.
- Split simulation work from observation construction for clearer profiling.

Expected payoff:

- Medium to high

Risk:

- Medium

### Version 0.5

Focus: replay and allocation reduction.

Features / tasks:

- Replace `copy.deepcopy(self.state)` replay capture with a replay-specific serializer.
- Reduce transient `dict`, `list`, and `set` allocations in hot loops.
- Reuse per-step buffers where practical.

Expected payoff:

- Medium

Risk:

- Medium

### Version 0.6

Focus: structural Python refactors.

Features / tasks:

- Flatten hot-path entity data into cheaper internal representations where justified.
- Reduce nested dataclass access in the hottest simulation loops.
- Keep external APIs stable while simplifying the sim kernel internally.

Expected payoff:

- Medium

Risk:

- Medium to high

### Version 0.7

Focus: compiled Python before Rust.

Features / tasks:

1. Try Cython for LOS, walkability, visibility, and combat helpers.
2. Evaluate mypyc for type-stable pure-Python modules.
3. Use Numba only if hot paths are converted to numeric array-oriented code.

Constraint:

- Only move isolated kernels first.
- Do not port content loading, replay serialization, RLlib adapters, or scenario configuration prematurely.

### Version 0.8

Focus: Rust only if Python and compiled-Python options are still insufficient.

Rust becomes justified if all of the following are true:

- Profiling still shows the simulation kernel dominates runtime after versions `0.2` through `0.7`.
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

1. Fill out feature bullets under each version as implementation decisions get locked in.
2. Generate a `py-spy` flamegraph from the 7-phase smoke run.
3. Start version `0.2` with occupancy caches.
4. Re-profile after each version increment.

## Done Criteria

- Smoke curriculum runtime is materially shorter.
- Replay-enabled smoke runs remain practical.
- Phase transitions no longer feel blocked by environment step time.
- We have before/after profiles showing where the win came from.
