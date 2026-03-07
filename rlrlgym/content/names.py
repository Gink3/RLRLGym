"""Race-aware agent name generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_name_table(path: str | Path = "data/names.json") -> Dict[str, Tuple[List[str], List[str]]]:
    p = Path(path)
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    names_raw = raw.get("names", {})
    if not isinstance(names_raw, dict):
        return {}
    out: Dict[str, Tuple[List[str], List[str]]] = {}
    for race, row in names_raw.items():
        if not isinstance(row, dict):
            continue
        first = [str(x).strip() for x in list(row.get("first_names", [])) if str(x).strip()]
        last = [str(x).strip() for x in list(row.get("last_names", [])) if str(x).strip()]
        if first and last:
            out[str(race).strip()] = (first, last)
    return out


def generate_full_name(
    names_by_race: Dict[str, Tuple[List[str], List[str]]],
    *,
    race: str,
    seed_key: str,
) -> str:
    race_key = str(race).strip()
    pool = names_by_race.get(race_key)
    if pool is None and names_by_race:
        pool = names_by_race.get("human") or next(iter(names_by_race.values()))
    if pool is None:
        return f"{race_key.title()} Wanderer".strip()
    first, last = pool
    digest = hashlib.sha256(seed_key.encode("utf-8")).digest()
    n0 = int.from_bytes(digest[:8], byteorder="big", signed=False)
    n1 = int.from_bytes(digest[8:16], byteorder="big", signed=False)
    return f"{first[n0 % len(first)]} {last[n1 % len(last)]}"
