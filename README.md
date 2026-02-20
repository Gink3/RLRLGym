# RLRLGym
RougeLike Reinforcement Learning Gym

## Minimal Running Example

Run from repo root:

```bash
python3 examples/minimal_run.py
python3 examples/window_demo.py
```

## Window Rendering

Open a dedicated render window (Tkinter):

```python
from rlrlgym import EnvConfig, PettingZooParallelRLRLGym

env = PettingZooParallelRLRLGym(EnvConfig(render_enabled=True))
env.reset(seed=7)
env.open_render_window()
for _ in range(20):
    env.step({"agent_0": 4, "agent_1": 4})
```

The window includes:
- `Play`, `Pause`, `Step` controls for playback frames
- fixed speed controls: `1x`, `2x`, `5x`
- `Focus` selector to center on a single agent
- `Zoom` slider in range `0..10` to zoom in/out around the selected agent
- tile colors rendered in the GUI window

Rendering is optional via `EnvConfig(render_enabled=False)`.
There is no CLI render mode.

## Run Tests

```bash
python3 -m unittest discover -s tests -q
```

## What Is Implemented

- PettingZoo Parallel-style multi-agent environment with `reset(seed, options)` / `step(actions)`
- Configurable per-agent observations
- JSON tile schema with required `schema_version` and required tile fields
- Reward shaping with interaction caps and anti-exploit penalties
- Window-only rendering with playback controls and focused zoom
- Snapshot save/load and synchronous vectorized environment wrapper
