# Ideas / Notes (RLRLGym)

## Multi-agent + Teams (RLlib)

- Having **many agents in one env** (e.g., 5 humans vs 5 orcs) is normal in RLlib multi-agent, but learning gets harder due to **non-stationarity**.
- Best practice for stability/sample-efficiency: **parameter sharing** within a class/team:
  - One shared policy for all humans
  - One shared policy for all orcs
  - For N teams: one policy per team (your stated plan)
- Keep **global step** semantics (all agents propose actions each tick). This generally scales better than strict AEC turn-by-turn.

## Reward Design (Team Deathmatch, N Teams)

### Separate “game objective” vs “style shaping”

1) **Shared objective** (symmetric across teams): win the deathmatch.
2) **Team-specific shaping**: encourage different styles (healing-heavy team, exploration-heavy team, etc.) while keeping win reward dominant.

### Terminal rewards

- For team deathmatch:
  - If team *t* is last surviving: `team_reward[t] += +W`
  - All other teams: `team_reward[others] += -W` (or `0`, but `-W` is sharper)
- Avoid giving a **positive timeout reward to everyone** (e.g., +10 on timeout), because it can incentivize stalling.
- Better:
  - Timeout with multiple teams alive: `0` or a **small negative** (e.g., `-T`) to discourage stalemates.

### Dense event rewards (to avoid sparse learning)

Keep these much smaller than W:
- On hit / damage: small positive
- On kill: medium positive, optionally split into team+individual components to improve credit assignment
- Optional per-step cost (`-c`) to reduce “wait it out” strategies

### Exploit-resistant shaping

- Exploration rewards are farmable: cap them (first-visit only, per-episode caps, etc.)
- Healing rewards are farmable: reward effective/meaningful healing; consider caps.

## Simultaneous Action Resolution (Movement Conflicts)

When multiple agents attempt to move into the same tile in one global step, choose a deterministic rule.

Common options:

1) **Bounce/block all**
- If conflict: no one moves.
- Simple/stable, but can jam.

2) **Priority winner**
- Choose one winner (fixed priority or random-but-seeded); losers stay.
- Less deadlock; fixed priority can bias.

3) **Allow swaps**
- If A wants B’s tile and B wants A’s tile, allow swap.
- For other multi-way conflicts, use (1) or (2).

Suggested PoC approach: **(3) swaps + (1) bounce** or **(3) swaps + random-seeded winner**.

## Action/Observation Space Divergence (Later)

- Plan: use **superset** action/obs space + **masking** so policies share spaces but don’t access forbidden actions/info.

### Action masking
- Use `action_space = Discrete(N_max)` for all.
- Provide an `action_mask` (0/1) per agent.
- RLlib typically expects the mask to be part of the **observation** (e.g., `{"obs": features, "action_mask": mask}`), not only in `info`.

### Observation masking
- Best: do not expose forbidden info in features (or zero it) and optionally include an `obs_mask`.
- Note: agents can still infer some hidden state indirectly through dynamics; that’s normal.

## “Classic Roguelike” Features to Add (Layered)

Ordered roughly from high impact / low complexity to more complex:

### 1) Combat with positional tactics
- Add an explicit **attack action** (melee adjacency) with damage.
- Add simple gear modifiers: weapon bonus / armor reduction.
- Log events: `hit`, `kill`, `damage` for reward shaping and analytics.

### 2) Doors + chokepoints (map topology)
- Closed doors block movement/LOS; `interact/open` toggles.
- Later: keys/locks.
- Mapgen: rooms + corridors to create ambushes and meaningful navigation.

### 3) Fog of war / line-of-sight
- Visibility blocked by walls.
- Only include enemies/items in observation when visible.
- Later: memory map.


### 5) Status effects
- Keep a small set at first: poison, bleed, stun, regen.


### 7) Objectives that force engagement
- Control points (shrines), loot rooms, or a shrinking safe zone to prevent hiding.

## Suggested Next Steps (Concrete)

Given existing systems (tiles, interactions, inventory, hunger):
1) Implement **attack + kill credit**
2) Add **doors/walls** in mapgen
3) Add **LOS-limited** observations


## Magic System (design)

### Action space
- Add a distinct **CAST** action id (separate from USE).
- Keep scroll learning under **USE**.

### Spells are knowledge (not items)
- Each agent has a list of known spells:
  - `known_spells: [spell_id, ...]`
- Casting consumes **MP** (mana) and triggers per-spell cooldowns (optional but recommended).

### MP / INT integration
- `MaxMP = MP_BASE + INT * MP_PER_INT` (tunable constants).
- INT also influences spell effectiveness (damage/heal scaling) and resistance (later).

### Scrolls teach spells
- Scroll items map to a spell id (e.g., `scroll_firebolt -> firebolt`).
- On **USE(scroll)**:
  - if spell not known: add to `known_spells`
  - consume scroll
  - emit event `learn_spell:<spell_id>` for rewards/analytics

### Spellbook items
- A **spellbook** is just gear that modifies INT (e.g., `INT +1/+2`).
- It does **not** contain spells.

### Starter vs found spells
- Classes can grant a small starter `known_spells` list.
- Stronger/situational spells are learned via scroll loot from chests/monsters.

---

## Scenario Builder (Tkinter UI) (design)

Goal: a UI to build scenarios with **N teams** and **K agents per team**.

### Scenario structure
- Each agent selects:
  - **race** (predefined list; current agents ≈ races)
  - **class** (predefined list)
  - starting skills/items (different per agent)
  - starting `known_spells` (not items)
- Each team has a shared:
  - **policy** (one NN/policy per team; parameter sharing within team)

### Required UI features
- Add/remove teams; set K and generate agents.
- Per-team policy editor (NN config) + checkpoint picker.
- Per-agent editor (race/class/skills/items/known_spells).
- Save/Load scenario files.
- "Resume training" workflow: pick scenario + optionally load checkpoints.
- Final review step: raw editable JSON panel with validate/apply (manual overrides).

### Optional: map generation in scenario builder
- Expose mapgen parameters + seed.
- Provide a map preview and regenerate button.
- Allow editing mapgen values directly in the scenario UI.

* Monsters need to be able to move and attack
* Agents should be referenced by their name and class
* replay viewer - zoom should not affect the turn log or the agent stats windows
* replay viewer - Should have a prev/next button to look for the next episode replay in the same directory