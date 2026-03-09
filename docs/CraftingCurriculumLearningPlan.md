# Crafting Curriculum Learning Plan (Toward 1000x1000)

## Objective

Train 10 agents to reliably gather, craft, and build in increasingly large worlds, ending with stable behavior on a `1000x1000` map.

## Strategy

Use staged curriculum maps and keep the task distribution craft-heavy early, then reintroduce full survival/combat pressure later.

- Early phases optimize task completion frequency.
- Mid phases stress logistics and long-horizon planning.
- Final phases validate robustness at very large scale.

## Training Assets

- Scenario: `data/scenarios/crafting_curriculum_10_agents`
- Curriculum config: `data/base/curriculum_phases_crafting_1000.json`

## Phased Map Progression

1. `32x32` bootstrap
- Goal: basic gathering and first successful construction loops.
- Environment: zero monster pressure, high chest density, short horizon.

2. `64x64` short logistics
- Goal: repeated kit crafting and structure placement.
- Environment: low threat, still dense resources.

3. `128x128` distributed resources
- Goal: route planning for multi-step recipes.
- Environment: low-medium threat, longer horizon.

4. `256x256` sparse travel
- Goal: maintain throughput despite travel distance.
- Environment: moderate threat, lower chest density.

5. `400x400` mixed pressure
- Goal: crafting under interruptions and partial failures.
- Environment: moderate threat, normal hunger pressure.

6. `700x700` long-horizon coordination
- Goal: sustain production chains over long episodes.
- Environment: moderate/high threat, stronger exploration penalties.

7. `1000x1000` target-scale validation
- Goal: stable construction/crafting behavior at final scale.
- Environment: near-target production settings and full pressure.

## 10-Agent Roster Design

Use a heterogeneous roster so learned behavior generalizes across stat templates:

- 5 explorer-leaning profiles (`human` profile)
- 5 brawler-leaning profiles (`orc` profile)
- Mixed races/classes to avoid overfitting to one body plan

## Metrics To Gate Phase Promotion

Promote when all gates hold for a rolling window:

- Construction success rate above threshold.
- Time-to-first-build below threshold.
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
