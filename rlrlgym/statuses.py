"""Status effect definitions and loading."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class StatusDef:
    status_id: str
    duration: int
    tick_interval: int = 1
    apply_effects: List[Dict[str, object]] = field(default_factory=list)
    tick_effects: List[Dict[str, object]] = field(default_factory=list)
    expire_effects: List[Dict[str, object]] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


def load_statuses(path: str | Path) -> Dict[str, StatusDef]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_statuses(raw)


def parse_statuses(raw: object) -> Dict[str, StatusDef]:
    if not isinstance(raw, dict):
        raise ValueError("Statuses JSON must be an object")
    if "schema_version" not in raw or not isinstance(raw["schema_version"], int):
        raise ValueError("Statuses JSON requires integer schema_version")
    if "statuses" not in raw or not isinstance(raw["statuses"], list):
        raise ValueError("Statuses JSON requires array 'statuses'")
    out: Dict[str, StatusDef] = {}
    for idx, row in enumerate(raw["statuses"]):
        if not isinstance(row, dict):
            raise ValueError(f"statuses[{idx}] must be an object")
        status_id = str(row.get("id", "")).strip()
        if not status_id:
            raise ValueError(f"statuses[{idx}] missing required field 'id'")
        out[status_id] = StatusDef(
            status_id=status_id,
            duration=max(1, int(row.get("duration", 1))),
            tick_interval=max(1, int(row.get("tick_interval", 1))),
            apply_effects=[
                dict(x) for x in row.get("apply", []) if isinstance(x, dict)
            ],
            tick_effects=[
                dict(x) for x in row.get("tick", []) if isinstance(x, dict)
            ],
            expire_effects=[
                dict(x) for x in row.get("expire", []) if isinstance(x, dict)
            ],
            tags=[str(x) for x in row.get("tags", []) if str(x).strip()],
        )
    return out
