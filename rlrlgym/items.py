"""Item definition loading and validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

DAMAGE_TYPE_SLASH = "slash"
DAMAGE_TYPE_PIERCE = "pierce"
DAMAGE_TYPE_BLUNT = "blunt"

DAMAGE_TYPES = {DAMAGE_TYPE_SLASH, DAMAGE_TYPE_PIERCE, DAMAGE_TYPE_BLUNT}
ARMOR_SLOTS = {
    "head",
    "chest",
    "back",
    "arms",
    "legs",
    "neck",
    "ring",
    "ring_1",
    "ring_2",
    "ring_3",
    "ring_4",
}
WEAPON_SKILLS = {"melee", "archery", "thrown_weapons"}


@dataclass
class WeaponDef:
    damage_type: str
    damage_min: int
    damage_max: int
    skill: str


@dataclass
class ItemDef:
    item_id: str
    weight: float
    edible_hunger: int = 0
    is_treasure: bool = False
    armor_slot: Optional[str] = None
    dr_bonus_vs: Dict[str, int] = field(default_factory=dict)
    weapon: Optional[WeaponDef] = None


@dataclass
class ItemCatalog:
    items: Dict[str, ItemDef]
    chest_loot_table: List[str]

    @property
    def edible_items(self) -> set[str]:
        return {item_id for item_id, item in self.items.items() if int(item.edible_hunger) > 0}

    @property
    def item_weight(self) -> Dict[str, float]:
        return {item_id: float(item.weight) for item_id, item in self.items.items()}

    @property
    def treasure_items(self) -> set[str]:
        return {
            item_id for item_id, item in self.items.items() if bool(item.is_treasure)
        }

    @property
    def armor_slot_by_item(self) -> Dict[str, str]:
        return {
            item_id: str(item.armor_slot)
            for item_id, item in self.items.items()
            if item.armor_slot is not None
        }

    @property
    def item_dr_bonus_vs(self) -> Dict[str, Dict[str, int]]:
        return {
            item_id: {k: int(v) for k, v in item.dr_bonus_vs.items()}
            for item_id, item in self.items.items()
            if item.dr_bonus_vs
        }

    @property
    def weapon_damage_type(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for item_id, item in self.items.items():
            if item.weapon is None:
                continue
            out[item_id] = str(item.weapon.damage_type)
        return out

    @property
    def weapon_damage_range(self) -> Dict[str, Tuple[int, int]]:
        out: Dict[str, Tuple[int, int]] = {}
        for item_id, item in self.items.items():
            if item.weapon is None:
                continue
            out[item_id] = (int(item.weapon.damage_min), int(item.weapon.damage_max))
        return out

    @property
    def weapon_skill(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for item_id, item in self.items.items():
            if item.weapon is None:
                continue
            out[item_id] = str(item.weapon.skill)
        return out


REQUIRED_ITEM_FIELDS = {"id", "weight"}


def load_items(path: str | Path) -> ItemCatalog:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Items JSON must be an object")
    if "schema_version" not in raw or not isinstance(raw["schema_version"], int):
        raise ValueError("Items JSON requires integer schema_version")
    if "items" not in raw or not isinstance(raw["items"], list):
        raise ValueError("Items JSON requires array 'items'")
    if "chest_loot_table" not in raw or not isinstance(raw["chest_loot_table"], list):
        raise ValueError("Items JSON requires array 'chest_loot_table'")

    items: Dict[str, ItemDef] = {}
    for idx, row in enumerate(raw["items"]):
        if not isinstance(row, dict):
            raise ValueError(f"item[{idx}] must be an object")
        missing = REQUIRED_ITEM_FIELDS - set(row.keys())
        if missing:
            miss = ", ".join(sorted(missing))
            raise ValueError(f"item[{idx}] missing required field(s): {miss}")

        item_id = str(row["id"])
        if item_id in items:
            raise ValueError(f"item[{idx}] duplicates id '{item_id}'")

        weight = float(row["weight"])
        edible_hunger = int(row.get("edible_hunger", 0))
        is_treasure = bool(row.get("is_treasure", False))

        armor_slot = row.get("armor_slot")
        if armor_slot is not None:
            armor_slot = str(armor_slot)
            if armor_slot not in ARMOR_SLOTS:
                raise ValueError(f"item[{idx}].armor_slot '{armor_slot}' must be one of {sorted(ARMOR_SLOTS)}")

        dr_bonus_vs_raw = row.get("dr_bonus_vs", {})
        if not isinstance(dr_bonus_vs_raw, dict):
            raise ValueError(f"item[{idx}].dr_bonus_vs must be an object")
        dr_bonus_vs: Dict[str, int] = {}
        for damage_type, bonus in dr_bonus_vs_raw.items():
            key = str(damage_type)
            if key not in DAMAGE_TYPES:
                raise ValueError(
                    f"item[{idx}].dr_bonus_vs has unknown damage type '{key}'"
                )
            dr_bonus_vs[key] = int(bonus)

        weapon_raw = row.get("weapon")
        weapon: Optional[WeaponDef] = None
        if weapon_raw is not None:
            if not isinstance(weapon_raw, dict):
                raise ValueError(f"item[{idx}].weapon must be an object")
            required_weapon_fields = {"damage_type", "damage_min", "damage_max", "skill"}
            wmissing = required_weapon_fields - set(weapon_raw.keys())
            if wmissing:
                miss = ", ".join(sorted(wmissing))
                raise ValueError(f"item[{idx}].weapon missing required field(s): {miss}")
            damage_type = str(weapon_raw["damage_type"])
            if damage_type not in DAMAGE_TYPES:
                raise ValueError(
                    f"item[{idx}].weapon.damage_type '{damage_type}' must be one of {sorted(DAMAGE_TYPES)}"
                )
            damage_min = int(weapon_raw["damage_min"])
            damage_max = int(weapon_raw["damage_max"])
            if damage_max < damage_min:
                raise ValueError(f"item[{idx}].weapon damage_max must be >= damage_min")
            skill = str(weapon_raw["skill"])
            if skill not in WEAPON_SKILLS:
                raise ValueError(
                    f"item[{idx}].weapon.skill '{skill}' must be one of {sorted(WEAPON_SKILLS)}"
                )
            weapon = WeaponDef(
                damage_type=damage_type,
                damage_min=damage_min,
                damage_max=damage_max,
                skill=skill,
            )

        items[item_id] = ItemDef(
            item_id=item_id,
            weight=weight,
            edible_hunger=edible_hunger,
            is_treasure=is_treasure,
            armor_slot=armor_slot,
            dr_bonus_vs=dr_bonus_vs,
            weapon=weapon,
        )

    if not items:
        raise ValueError("At least one item is required")

    chest_loot_table: List[str] = []
    for idx, item_id in enumerate(raw["chest_loot_table"]):
        key = str(item_id)
        if key not in items:
            raise ValueError(
                f"chest_loot_table[{idx}] references unknown item '{key}'"
            )
        chest_loot_table.append(key)

    return ItemCatalog(items=items, chest_loot_table=chest_loot_table)
