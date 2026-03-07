"""Spell definitions and loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class SpellDef:
    spell_id: str
    mana_cost: int = 0
    cooldown: int = 0
    range: int = 0
    target: str = "self"
    effects: List[Dict[str, object]] = field(default_factory=list)
    required_reagents: Dict[str, int] = field(default_factory=dict)


def load_spells(path: str | Path) -> Dict[str, SpellDef]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_spells(raw)


def parse_spells(raw: object) -> Dict[str, SpellDef]:
    if not isinstance(raw, dict):
        raise ValueError("Spells JSON must be an object")
    if "schema_version" not in raw or not isinstance(raw["schema_version"], int):
        raise ValueError("Spells JSON requires integer schema_version")
    if "spells" not in raw or not isinstance(raw["spells"], list):
        raise ValueError("Spells JSON requires array 'spells'")
    out: Dict[str, SpellDef] = {}
    for idx, row in enumerate(raw["spells"]):
        if not isinstance(row, dict):
            raise ValueError(f"spells[{idx}] must be an object")
        spell_id = str(row.get("id", "")).strip()
        if not spell_id:
            raise ValueError(f"spells[{idx}] missing required field 'id'")
        out[spell_id] = SpellDef(
            spell_id=spell_id,
            mana_cost=max(0, int(row.get("mana_cost", 0))),
            cooldown=max(0, int(row.get("cooldown", 0))),
            range=max(0, int(row.get("range", 0))),
            target=str(row.get("target", "self")).strip() or "self",
            effects=[dict(x) for x in row.get("effects", []) if isinstance(x, dict)],
            required_reagents={
                str(k): max(1, int(v))
                for k, v in dict(row.get("required_reagents", {})).items()
            },
        )
    return out
