"""Map structure generation config loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_structures_config(path: str | Path) -> List[Dict[str, object]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_structures_config(raw)


def parse_structures_config(raw: object) -> List[Dict[str, object]]:
    if not isinstance(raw, dict):
        raise ValueError("Structures config JSON must be an object")
    if "schema_version" not in raw or not isinstance(raw["schema_version"], int):
        raise ValueError("Structures config JSON requires integer schema_version")
    rows = raw.get("structures", [])
    if not isinstance(rows, list):
        raise ValueError("Structures config JSON requires array 'structures'")
    return [dict(x) for x in rows if isinstance(x, dict)]
