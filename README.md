# RLRLGym
RougeLike Reinforcement Learning Gym

## Minimal Running Example

Run from repo root:

```bash
python3 examples/minimal_run.py
python3 examples/window_demo.py
python3 examples/training_dashboard_demo.py
python3 examples/train_demo.py
python3 -m train --episodes 100 --max-steps 120 --output-dir outputs/train
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

## Observation And Action Spaces

The environment exposes per-agent spaces in PettingZoo Parallel style:

- `env.action_space(agent_id)` returns a discrete integer range `(0, 10)`
- `env.observation_space(agent_id)` returns a dict-style shape descriptor based on the agent profile

### Action Space

Each action is an integer in `0..10`:

- `0`: move north
- `1`: move south
- `2`: move west
- `3`: move east
- `4`: wait/rest
- `5`: loot
- `6`: eat
- `7`: pick up items
- `8`: equip item
- `9`: use item
- `10`: interact with environment / nearby agent

### Observation Space

Observations are per-agent dictionaries and always include:

- `step`: current environment step
- `alive`: whether the agent is alive
- `profile`: profile name (for example `human`, `orc`)

Profile and config determine optional keys:

- `local_tiles`: local tile window around the agent (`view_radius` dependent)
- `stats`: `{hp, hunger, position, equipped_count}`
- `inventory`: list of carried items

Example:

```python
obs, info = env.reset(seed=7)
agent_obs = obs["agent_0"]
# agent_obs -> {"step": 0, "alive": True, "profile": "human", ...}
```

## Train Module

The in-repo `train/` module provides a neural Q-learning baseline over
the PettingZoo-style env.

CLI:

```bash
python3 -m train --episodes 100 --max-steps 120 --seed 0 --output-dir outputs/train --networks-path data/agent_networks.json
```

Outputs include:
- `outputs/train/neural_policies.json` (learned neural policies)
- dashboard artifacts via `TrainingLogger`

Network architectures are defined in `data/agent_networks.json` by profile name
(for example `human` and `orc`).

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
