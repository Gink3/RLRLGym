"""Helpers for archiving training inputs into run output directories."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, Optional


def _copy_path(src: Path, dst: Path) -> None:
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def archive_training_inputs(
    *,
    output_dir: str,
    env_config_path: str,
    networks_path: Optional[str] = None,
    curriculum_path: Optional[str] = None,
    scenario_path: Optional[str] = None,
) -> Dict[str, str]:
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    snapshot_root = out / "data_snapshot"
    snapshot_root.mkdir(parents=True, exist_ok=True)

    copied: Dict[str, str] = {}

    data_root = Path("data")
    if data_root.exists():
        dst = snapshot_root / "data"
        _copy_path(data_root, dst)
        copied["data"] = str(dst)

    def copy_optional(label: str, raw: Optional[str]) -> None:
        if not raw:
            return
        src = Path(str(raw))
        if not src.exists():
            return
        if data_root.exists():
            try:
                src_res = src.resolve()
                data_res = data_root.resolve()
                if src_res.is_relative_to(data_res):  # type: ignore[attr-defined]
                    return
            except Exception:
                pass
        dst = snapshot_root / "external_inputs" / label / src.name
        _copy_path(src, dst)
        copied[label] = str(dst)

    copy_optional("env_config", env_config_path)
    copy_optional("networks", networks_path)
    copy_optional("curriculum", curriculum_path)
    copy_optional("scenario", scenario_path)

    manifest = {
        "schema_version": 1,
        "snapshot_root": str(snapshot_root),
        "source_paths": {
            "env_config_path": str(env_config_path),
            "networks_path": str(networks_path or ""),
            "curriculum_path": str(curriculum_path or ""),
            "scenario_path": str(scenario_path or ""),
        },
        "copied": copied,
    }
    manifest_path = snapshot_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    copied["manifest"] = str(manifest_path)
    return copied
