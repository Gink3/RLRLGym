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
from rlrlgym import EnvConfig, MultiAgentRLRLGym

env = MultiAgentRLRLGym(EnvConfig(render_enabled=True))
env.reset(seed=7)
env.open_render_window()
for _ in range(20):
    env.step({"agent_0": 4, "agent_1": 4})
```

The window includes:
- `Play`, `Pause`, `Step`, `Fast x2` controls for playback frames
- `Focus` selector to center on a single agent
- `Zoom` slider to zoom in/out around the selected agent

Rendering is optional via `EnvConfig(render_enabled=False)`.

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
