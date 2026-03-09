# Core Systems Curriculum Learning Plan (Toward 1000x1000)

## Objective

Train 10 agents to reliably use all core systems in increasingly large worlds, ending with stable behavior on a `1000x1000` map.

Core systems targeted:

- survival (hunger/exploration pressure)
- gathering and crafting
- construction and workstation economy
- station-based smelting/minting
- animals/ecology interaction
- combat + team coordination

## Strategy

Use staged curriculum maps and gradually expand from low-risk economy tasks to full-system pressure.

- Early phases optimize survival + gather/craft completion frequency.
- Mid phases stress construction, station economy, and logistics.
- Late phases stress combat, coordination, and long-horizon execution.
- Final phase validates all systems at target scale.

## Training Assets

- Scenario: `data/scenarios/crafting_curriculum_10_agents`
- Curriculum config: `data/base/curriculum_phases_crafting_1000.json`

## Phased Map Progression

1. `32x32` survival and gathering bootstrap
- Goal: reliable exploration + resource acquisition loops.
- Environment: zero monster pressure, high chest density, short horizon.

2. `64x64` crafting and construction
- Goal: repeated kit crafting + structure placement.
- Environment: low threat, still dense resources.

3. `128x128` station economy
- Goal: gather -> smelt -> mint pipeline reliability.
- Environment: low-medium threat, longer horizon.

4. `256x256` animals and resource pressure
- Goal: maintain production with ecology competition and travel costs.
- Environment: moderate threat, lower chest density.

5. `400x400` team coordination
- Goal: sustain crafting/economy while coordinating movement under pressure.
- Environment: moderate threat, normal hunger pressure.

6. `700x700` combat and system pressure
- Goal: maintain economy and survival while handling combat disruption.
- Environment: moderate/high threat, stronger exploration penalties.

7. `1000x1000` full-system validation
- Goal: stable integrated behavior across all core systems at final scale.
- Environment: near-target production settings and full pressure.

## 10-Agent Roster Design

Use a heterogeneous roster so learned behavior generalizes across stat templates:

- 5 explorer-leaning profiles (`human`)
- 5 brawler-leaning profiles (`orc`)
- Mixed races/classes to avoid overfitting to one body plan

## Metrics To Gate Phase Promotion

Promote when all gates hold for a rolling window:

- Construction success rate above threshold.
- Time-to-first-build below threshold.
- Station economy completion rate (smelt + mint chain).
- Combat engagement and survival stability.
- Stable mean survival time.
- Non-trivial exploration coverage.

If a phase fails gates, keep map size fixed and tune reward/curriculum settings before escalating map dimensions.

## Initial Run Command

```bash
python3 -m train \
  --backend rllib \
  --scenario-path data/scenarios/crafting_curriculum_10_agents \
  --curriculum-path data/base/curriculum_phases_crafting_1000.json \
  --iterations 200 \
  --algo ppo_masked \
  --shared-policy \
  --output-dir outputs/train/crafting_curriculum_10_agents
```

## Notes

- Use this as a baseline plan; phase durations should be adjusted with observed metrics.
- Keep scenario constant while scaling maps to isolate curriculum effects.
