# RLRL Gym


A gym to simulate brogue/nethack/Cataclysm DDA.


# High Level Design
* The game model should be expandable in order to add new tiles, items, and monsters
* The game model needs a gym-compatible api.
* Observations should be configurable per agent to make asymmetric experiences.
* There should be a way to view/render training or playback.
    * The rendering needs to support tile sets eventually, but colored ascii tiles for now is fine
    * The renderings need to have pause/play/fast forward capability
    * The renderings need to be able to zoom in and out to focus on single agents

## Generic Actions
* Move
* Wait/Rest
* Loot
* Eat
* Pick up items
* Equip Items
* Use Items
* Interact with the environment


## Rewards
Reward Design needs to reward interactions with the enviornment and other agents. To limit exploitation, the environment needs to limit how many times something can be interacted with. Carefully shaped auxiliary rewards (exploration, skill gain, resource management) to avoid exploit loops. Add penalties for degenerate behavior (stutter-step farming, infinite wait loops).


## Procedural Generation
The map generation engine should be configurable to allow for different generation probabilities. All tiles should be defined in json. All maps are assumed to be on the same z level or 1 z level. 


## Performance + scaling
Vectorized environments (many parallel instances).
Profile hotspots; move heavy simulation logic to optimized code if needed.
Save/load snapshots for fast evaluation and debugging.

## Testing + reproducibility
Unit tests for core mechanics.
Property tests for invariants (no illegal state transitions).
Fixed-seed regression tests for training/eval consistency.

## Baselines + training stack
Start with PPO/A2C on simplified rules.
Add recurrent policies (LSTM/Transformer) for partial observability.
Log rich metrics (survival time, dungeon depth, cause of death, resource efficiency).
