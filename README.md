# RLRLGym
RougeLike Reinforcement Learning Gym

## Minimal Running Example

Run from repo root:

```bash
python3 examples/minimal_run.py
python3 examples/window_demo.py
python3 examples/training_dashboard_demo.py
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

## Training Logger Dashboard

Generate aggregate training metrics and dashboard artifacts:

```bash
python3 examples/training_dashboard_demo.py
```

Outputs are written to `outputs/`:
- `episodes.jsonl`
- `episodes.csv`
- `summary.json`
- `dashboard.html`

## Run Tests

```bash
python3 -m unittest discover -s tests -q
```

## What Is Implemented

- PettingZoo Parallel-style multi-agent environment with `reset(seed, options)` / `step(actions)`
- Configurable per-agent observations
- Agent profile system loaded from `data/agent_profiles.json` with `human` and `orc` defaults (different observations and reward shaping)
- JSON tile schema with required `schema_version` and required tile fields
- Reward shaping with interaction caps and anti-exploit penalties
- Window-only rendering with playback controls and focused zoom
- Dedicated training logger/dashboard artifacts (`episodes.jsonl`, `episodes.csv`, `summary.json`, `dashboard.html`)
- Snapshot save/load and synchronous vectorized environment wrapper
