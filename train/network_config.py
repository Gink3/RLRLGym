"""Neural network architecture config loading for training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class NetworkConfig:
    name: str
    hidden_layers: List[int]
    activation: str = "relu"
    learning_rate: float = 0.003
    gamma: float = 0.97
    epsilon_start: float = 0.2
    epsilon_end: float = 0.03
    epsilon_decay: float = 0.995


REQUIRED_ARCH_FIELDS = {
    "name",
    "hidden_layers",
    "activation",
    "learning_rate",
    "gamma",
    "epsilon_start",
    "epsilon_end",
    "epsilon_decay",
}


def load_network_configs(path: str | Path) -> Dict[str, NetworkConfig]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Network config JSON must be an object")
    if "schema_version" not in raw or not isinstance(raw["schema_version"], int):
        raise ValueError("Network config JSON requires integer schema_version")
    if "architectures" not in raw or not isinstance(raw["architectures"], list):
        raise ValueError("Network config JSON requires array 'architectures'")

    out: Dict[str, NetworkConfig] = {}
    for idx, row in enumerate(raw["architectures"]):
        if not isinstance(row, dict):
            raise ValueError(f"architectures[{idx}] must be an object")
        missing = REQUIRED_ARCH_FIELDS - set(row.keys())
        if missing:
            miss = ", ".join(sorted(missing))
            raise ValueError(f"architectures[{idx}] missing required field(s): {miss}")

        hidden = row["hidden_layers"]
        if not isinstance(hidden, list) or not hidden:
            raise ValueError(f"architectures[{idx}].hidden_layers must be a non-empty array")

        cfg = NetworkConfig(
            name=str(row["name"]),
            hidden_layers=[int(x) for x in hidden],
            activation=str(row["activation"]),
            learning_rate=float(row["learning_rate"]),
            gamma=float(row["gamma"]),
            epsilon_start=float(row["epsilon_start"]),
            epsilon_end=float(row["epsilon_end"]),
            epsilon_decay=float(row["epsilon_decay"]),
        )
        out[cfg.name] = cfg

    if not out:
        raise ValueError("At least one network architecture is required")
    return out
