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
PROFILE_VOCAB = ["human", "orc"]
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

    skills = stats.get("skills", {})
    if not isinstance(skills, dict):
        skills = {}
    skill_features = [float(skills.get(k, 0.0)) / 20.0 for k in SKILL_VOCAB]

    profile = str(obs.get("profile", "human"))
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
    ] + skill_features + profile_features + tile_features + item_features


def observation_vector_size() -> int:
    # 14 scalar stats + skill features + profile one-hot + tile histogram + item histogram
    return (
        14 + len(SKILL_VOCAB) + len(PROFILE_VOCAB) + len(TILE_VOCAB) + len(ITEM_VOCAB)
    )
