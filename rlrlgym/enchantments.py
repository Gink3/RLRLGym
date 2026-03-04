"""Enchantment definition loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class EnchantDef:
    enchant_id: str
    tags: List[str] = field(default_factory=list)
    effects: List[Dict[str, object]] = field(default_factory=list)
    max_stacks: int = 1


def load_enchantments(path: str | Path) -> Dict[str, EnchantDef]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_enchantments(raw)


def parse_enchantments(raw: object) -> Dict[str, EnchantDef]:
    if not isinstance(raw, dict):
        raise ValueError("Enchantments JSON must be an object")
    if "schema_version" not in raw or not isinstance(raw["schema_version"], int):
        raise ValueError("Enchantments JSON requires integer schema_version")
    if "enchantments" not in raw or not isinstance(raw["enchantments"], list):
        raise ValueError("Enchantments JSON requires array 'enchantments'")
    out: Dict[str, EnchantDef] = {}
    for idx, row in enumerate(raw["enchantments"]):
        if not isinstance(row, dict):
            raise ValueError(f"enchantments[{idx}] must be an object")
        enchant_id = str(row.get("id", "")).strip()
        if not enchant_id:
            raise ValueError(f"enchantments[{idx}] missing required field 'id'")
        out[enchant_id] = EnchantDef(
            enchant_id=enchant_id,
            tags=[str(x) for x in row.get("tags", []) if str(x).strip()],
            effects=[dict(x) for x in row.get("effects", []) if isinstance(x, dict)],
            max_stacks=max(1, int(row.get("max_stacks", 1))),
        )
    return out
