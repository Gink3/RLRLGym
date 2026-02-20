# RLRLGym
RougeLike Reinforcement Learning Gym

## Minimal Running Example

Run from repo root:

```bash
python3 examples/minimal_run.py
```

## Run Tests

```bash
python3 -m unittest discover -s tests -q
```

## What Is Implemented

- Multi-agent Gym-like environment with `reset` / `step`
- Configurable per-agent observations
- JSON-defined tiles and weighted procedural map generation
- Reward shaping with interaction caps and anti-exploit penalties
- ASCII rendering with playback controls (`pause`, `play`, `fast_forward`, `zoom` via focused render)
- Snapshot save/load and synchronous vectorized environment wrapper
