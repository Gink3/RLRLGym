"""Crafting recipe loading and validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class RecipeDef:
    recipe_id: str
    inputs: Dict[str, int]
    outputs: Dict[str, int]
    skill: str = "crafting"
    min_skill: int = 0
    station: str = ""
    craft_time: int = 1
    build_tile_id: str = ""
    required_tool_category: str = ""
    speed_multiplier: float = 1.0
    quality_bonus: float = 0.0
    tags: List[str] = field(default_factory=list)


def load_recipes(path: str | Path) -> Dict[str, RecipeDef]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return parse_recipes(raw)


def parse_recipes(raw: object) -> Dict[str, RecipeDef]:
    if not isinstance(raw, dict):
        raise ValueError("Recipes JSON must be an object")
    if "schema_version" not in raw or not isinstance(raw["schema_version"], int):
        raise ValueError("Recipes JSON requires integer schema_version")
    if "recipes" not in raw or not isinstance(raw["recipes"], list):
        raise ValueError("Recipes JSON requires array 'recipes'")

    out: Dict[str, RecipeDef] = {}
    for idx, row in enumerate(raw["recipes"]):
        if not isinstance(row, dict):
            raise ValueError(f"recipes[{idx}] must be an object")
        recipe_id = str(row.get("id", "")).strip()
        if not recipe_id:
            raise ValueError(f"recipes[{idx}] missing required field 'id'")
        if recipe_id in out:
            raise ValueError(f"recipes[{idx}] duplicates id '{recipe_id}'")

        inputs_raw = row.get("inputs", {})
        outputs_raw = row.get("outputs", {})
        if not isinstance(inputs_raw, dict) or not inputs_raw:
            raise ValueError(f"recipes[{idx}].inputs must be a non-empty object")
        if not isinstance(outputs_raw, dict):
            raise ValueError(f"recipes[{idx}].outputs must be an object")
        build_tile_id = str(row.get("build_tile_id", "")).strip()
        required_tool_category = str(row.get("required_tool_category", "")).strip().lower()
        if not outputs_raw and not build_tile_id:
            raise ValueError(
                f"recipes[{idx}] must define non-empty outputs or build_tile_id"
            )

        inputs = {str(k): max(1, int(v)) for k, v in inputs_raw.items()}
        outputs = {str(k): max(1, int(v)) for k, v in outputs_raw.items()}
        out[recipe_id] = RecipeDef(
            recipe_id=recipe_id,
            inputs=inputs,
            outputs=outputs,
            skill=str(row.get("skill", "crafting")),
            min_skill=max(0, int(row.get("min_skill", 0))),
            station=str(row.get("station", "")).strip(),
            craft_time=max(1, int(row.get("craft_time", 1))),
            build_tile_id=build_tile_id,
            required_tool_category=required_tool_category,
            speed_multiplier=max(0.1, float(row.get("speed_multiplier", 1.0))),
            quality_bonus=float(row.get("quality_bonus", 0.0)),
            tags=[str(x) for x in row.get("tags", []) if str(x).strip()],
        )
    return out
