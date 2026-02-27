"""Observation featurization utilities shared by trainers/adapters."""

from __future__ import annotations

from typing import Dict, List

TILE_VOCAB = [
    "floor",
    "wall",
    "water",
    "food_cache",
    "chest",
    "shrine",
    "void",
]
PROFILE_VOCAB = ["reward_explorer_policy_v1", "reward_brawler_policy_v1"]
ITEM_VOCAB = [
    "ration",
    "fruit",
    "bandage",
    "healing_potion",
    "antidote",
    "dagger",
    "short_sword",
    "long_sword",
    "spear",
    "club",
    "mace",
    "bow",
    "crossbow",
    "thrown_rock",
    "thrown_knife",
    "torch",
    "unknown",
]
SKILL_VOCAB = [
    "melee",
    "archery",
    "thrown_weapons",
    "medic",
    "athletics",
    "exploration",
    "armor_head",
    "armor_chest",
    "armor_back",
    "armor_arms",
    "armor_legs",
    "armor_neck",
    "armor_rings",
]
MONSTER_VOCAB = [
    "rat",
    "bat",
    "kobold_skirmisher",
    "goblin_scout",
    "skeleton",
    "giant_spider",
    "cult_acolyte",
    "armored_beetle",
    "unknown",
]


def vectorize_observation(obs: Dict[str, object]) -> List[float]:
    stats = obs.get("stats", {}) if isinstance(obs.get("stats"), dict) else {}
    hp = float(stats.get("hp", 0.0)) / 50.0
    hunger = float(stats.get("hunger", 0.0)) / 50.0
    equipped = float(stats.get("equipped_count", 0.0)) / 12.0
    inventory = obs.get("inventory", []) if isinstance(obs.get("inventory"), list) else []
    inventory_len = float(len(inventory)) / 20.0

    teammate_distance_raw = stats.get("teammate_distance")
    teammate_distance = (
        0.0 if teammate_distance_raw is None else float(teammate_distance_raw) / 50.0
    )
    strength = float(stats.get("strength", 5.0)) / 20.0
    dexterity = float(stats.get("dexterity", 5.0)) / 20.0
    intellect = float(stats.get("intellect", 5.0)) / 20.0
    overall_level = float(stats.get("overall_level", 0.0)) / 50.0
    encumbrance_ratio = float(stats.get("encumbrance_ratio", 0.0)) / 2.0

    tile_interactions = stats.get("tile_interaction_counts", {})
    if not isinstance(tile_interactions, dict):
        tile_interactions = {}
    current_tile_used = float(tile_interactions.get("current_tile_used", 0.0)) / 10.0
    total_used = float(tile_interactions.get("total_used", 0.0)) / 100.0

    nearby_chests = stats.get("nearby_chests", {})
    if not isinstance(nearby_chests, dict):
        nearby_chests = {}
    nearby_chests_closed = float(nearby_chests.get("closed", 0.0)) / 30.0
    nearby_chests_opened = float(nearby_chests.get("opened", 0.0)) / 30.0

    nearby_agents = stats.get("nearby_agents", {})
    if not isinstance(nearby_agents, dict):
        nearby_agents = {}
    nearby_agents_visible = float(nearby_agents.get("visible", 0.0)) / 8.0
    nearby_agents_adjacent = float(nearby_agents.get("adjacent", 0.0)) / 4.0
    nearby_agents_nearest_raw = nearby_agents.get("nearest_distance")
    nearby_agents_nearest = (
        1.0
        if nearby_agents_nearest_raw is None
        else min(1.0, float(nearby_agents_nearest_raw) / 50.0)
    )

    nearby_monsters = stats.get("nearby_monsters", {})
    if not isinstance(nearby_monsters, dict):
        nearby_monsters = {}
    nearby_monsters_visible = float(nearby_monsters.get("visible", 0.0)) / 20.0
    nearby_monsters_adjacent = float(nearby_monsters.get("adjacent", 0.0)) / 4.0
    nearby_monsters_nearest_raw = nearby_monsters.get("nearest_distance")
    nearby_monsters_nearest = (
        1.0
        if nearby_monsters_nearest_raw is None
        else min(1.0, float(nearby_monsters_nearest_raw) / 50.0)
    )

    skills = stats.get("skills", {})
    if not isinstance(skills, dict):
        skills = {}
    skill_features = [float(skills.get(k, 0.0)) / 20.0 for k in SKILL_VOCAB]

    profile = str(obs.get("profile", "reward_explorer_policy_v1"))
    profile_features = [1.0 if profile == p else 0.0 for p in PROFILE_VOCAB]

    tile_hist = {k: 0.0 for k in TILE_VOCAB}
    local = obs.get("local_tiles", [])
    if isinstance(local, list):
        for row in local:
            if not isinstance(row, list):
                continue
            for tile in row:
                tid = str(tile)
                if tid in tile_hist:
                    tile_hist[tid] += 1.0
                else:
                    tile_hist["void"] += 1.0

    total_tiles = max(1.0, sum(tile_hist.values()))
    tile_features = [tile_hist[k] / total_tiles for k in TILE_VOCAB]

    nearby_item_counts = stats.get("nearby_item_counts", {})
    if not isinstance(nearby_item_counts, dict):
        nearby_item_counts = {}
    item_counts = {k: 0.0 for k in ITEM_VOCAB}
    for item, count in nearby_item_counts.items():
        key = item if item in item_counts else "unknown"
        item_counts[key] += float(count)
    total_items = max(1.0, sum(item_counts.values()))
    item_features = [item_counts[k] / total_items for k in ITEM_VOCAB]

    monster_by_type = nearby_monsters.get("by_type", {})
    if not isinstance(monster_by_type, dict):
        monster_by_type = {}
    monster_counts = {k: 0.0 for k in MONSTER_VOCAB}
    for monster_id, count in monster_by_type.items():
        key = monster_id if monster_id in monster_counts else "unknown"
        monster_counts[key] += float(count)
    total_monsters = max(1.0, sum(monster_counts.values()))
    monster_features = [monster_counts[k] / total_monsters for k in MONSTER_VOCAB]

    return [
        hp,
        hunger,
        equipped,
        inventory_len,
        teammate_distance,
        strength,
        dexterity,
        intellect,
        overall_level,
        encumbrance_ratio,
        current_tile_used,
        total_used,
        nearby_chests_closed,
        nearby_chests_opened,
        nearby_agents_visible,
        nearby_agents_adjacent,
        nearby_agents_nearest,
        nearby_monsters_visible,
        nearby_monsters_adjacent,
        nearby_monsters_nearest,
    ] + skill_features + profile_features + tile_features + item_features + monster_features


def observation_vector_size() -> int:
    # 20 scalar stats + skill features + profile one-hot + tile histogram + item histogram + monster histogram
    return (
        20
        + len(SKILL_VOCAB)
        + len(PROFILE_VOCAB)
        + len(TILE_VOCAB)
        + len(ITEM_VOCAB)
        + len(MONSTER_VOCAB)
    )
