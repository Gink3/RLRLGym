# RLRLGym
RougeLike Reinforcement Learning Gym

## Terminology

- `Agent`: a single controllable unit in the environment (for example `agent_0`), with inventory, skills, health, hunger, faction state, race, and class.
- `Race`: the base stat template for an agent (strength/dexterity/intellect and damage-resistance traits) loaded from `data/base/agent_races.json`.
- `Class`: the starting role package for an agent (starting items and initial skill modifiers) loaded from `data/base/agent_classes.json`.
- `Scenario`: a runnable setup file containing `env_config` values plus an explicit list of agents (race/class/profile/network choices), typically stored in `data/scenarios/`.

## Minimal Running Example

Run from repo root:

```bash
python3 examples/minimal_run.py
python3 examples/window_demo.py
python3 examples/train_demo.py
./scripts/train_default.sh
```

## Tools

- `tools/view_replay.py`: replay viewer for `*.replay.json` files with playback controls, focus, zoom, and render mode switch (`ascii` default, optional `tileset`).
- `tools/scenario_editor.py`: GUI editor for scenario files (snapshot `env_config` + agent list). Agent creation flow is race + class selection followed by editable combined JSON before save.
- `tools/train_launcher.py`: GUI launcher for training jobs with live log streaming and basic live metrics (`return`, `win`, `survival`, `starvation`, `loss`, `epsilon`).
- `python3 -m train`: training CLI (custom and RLlib backends) with scenario support.
- Both GUI tools include a top-bar `Settings -> Theme` selector. Selected theme is shared and persisted in `data/user/tool_settings.json`.

Replay viewer example:

```bash
python3 tools/view_replay.py outputs/train/<run>/replays/latest_episode.replay.json
```

Scenario Editor example:

```bash
python3 tools/scenario_editor.py --scenario data/scenarios/all_race_class_combinations
```

Training Launcher example:

```bash
python3 tools/train_launcher.py
```

## Data Layout

- `data/base/`: stable shared game data used across scenarios (tiles, items, races, classes, profiles, monsters, network defaults, curriculum defaults).
- `data/scenarios/`: scenario-specific directories. Each scenario directory contains:
- `env_config.json`: full environment config for that scenario.
- `agents.json`: list of agents used in that scenario.
- `data/env_config.json`: active environment config entrypoint; points to base data by default and may be overridden per scenario.

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

## Aim UI

When running Aim locally, the UI is available on localhost port `43800`:

```text
http://127.0.0.1:43800
```

By default training logs Aim runs to `/proj/aimml`. Start the UI against that repo:

```bash
.venv/bin/aim up --repo /proj/aimml --host 127.0.0.1 --port 43800
```

## Observation And Action Spaces

The environment exposes per-agent spaces in PettingZoo Parallel style:

- `env.action_space(agent_id)` returns a discrete integer range `(0, 17)`
- `env.observation_space(agent_id)` returns a dict-style shape descriptor based on the agent profile

### Action Space

Each action is an integer in `0..17`:

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
- `11`: attack
- `12`: create faction / invite ally (contextual)
- `13`: give item to adjacent ally
- `14`: trade with adjacent ally
- `15`: revive adjacent ally
- `16`: guard adjacent ally
- `17`: accept pending faction invite

### Observation Space

Observations are per-agent dictionaries and always include:

- `step`: current environment step
- `alive`: whether the agent is alive
- `profile`: profile name (for example `reward_explorer_policy_v1`, `reward_brawler_policy_v1`)

Profile and config determine optional keys:

- `local_tiles`: local tile window around the agent (`view_width`/`view_height` dependent)
- `stats`: `{hp, hunger, position, equipped_count}`
- `inventory`: list of carried items

Example:

```python
obs, info = env.reset(seed=7)
agent_obs = obs["agent_0"]
# agent_obs -> {"step": 0, "alive": True, "profile": "reward_explorer_policy_v1", ...}
```

## Train Module

The in-repo `train/` module supports:

- `rllib` backend (primary, recommended)
- `custom` backend (legacy in-repo trainer)

CLI:

```bash
./scripts/train_default.sh
```

Outputs include:
- RLlib metrics/checkpoints in the selected output directory
- (custom backend only) `neural_policies.json` checkpoint

Network architectures are defined in `data/base/agent_networks.json` by profile name
(for example `default`, and optionally per-profile variants).

Install RLlib:

```bash
python3 -m pip install "ray[rllib]"
```

Direct RLlib CLI example:

```bash
python3 -m train --backend rllib --iterations 50 --max-steps 120 --output-dir outputs/train/default
```

Legacy custom backend example:

```bash
python3 -m train --backend custom --episodes 100 --max-steps 120 --output-dir outputs/train/custom --networks-path data/base/agent_networks.json
```

Scenario-driven custom training (one NN per agent in scenario roster):

```bash
python3 -m train --backend custom --scenario-path data/scenarios/all_race_class_combinations --output-dir outputs/train/scenario_run
```

NN capacity guard options:

- `--max-nn-policies <N>` hard cap.
- `--resource-guard-ram-fraction <f>` RAM fraction used for estimated cap (default `0.45`).
- `--resource-guard-bytes-per-param <b>` memory estimate per parameter (default `32`).
- `--no-resource-guard` disables the guard.

Version-controlled training scripts:

- `scripts/train_quick.sh` (fast RLlib training)
- `scripts/train_default.sh` (default RLlib training)
- `scripts/train_long.sh` (longer RLlib training)
- `scripts/train_full.sh` (full RLlib training)

All scripts accept additional CLI overrides, for example:

```bash
./scripts/train_quick.sh --seed 3 --output-dir outputs/train/custom_run
```

Training metrics are also logged to Aim (when `aim` is installed), including dashboard-equivalent episode/iteration metrics for both `custom` and `rllib` backends.
Use `--aim-experiment <name>` to change the experiment, `--aim-repo <path>` to change repo location, and `--no-aim` to disable Aim logging.

## Run Tests

```bash
python3 -m unittest discover -s tests -q
```

## What Is Implemented

- PettingZoo Parallel-style multi-agent environment with `reset(seed, options)` / `step(actions)`
- Configurable per-agent observations
- Agent profile system loaded from `data/base/agent_profiles.json` with descriptive reward/network profile names
- JSON tile schema with required `schema_version` and required tile fields
- Reward shaping with interaction caps and anti-exploit penalties
- Window-only rendering with playback controls and focused zoom
- Aim-native training metrics logging for both backends
- Snapshot save/load and synchronous vectorized environment wrapper
