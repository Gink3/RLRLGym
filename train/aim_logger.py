"""Lightweight optional Aim metric logging helpers."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable


class AimLogger:
    def __init__(
        self,
        enabled: bool,
        experiment: str,
        output_dir: str,
        run_name: str | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self._run = None
        if not self.enabled:
            return
        try:
            from aim import Run
        except Exception:
            self.enabled = False
            return
        repo = str(Path(output_dir).resolve())
        self._run = Run(repo=repo, experiment=str(experiment))
        if run_name:
            self._run.name = str(run_name)

    def set_params(self, params: Dict[str, object]) -> None:
        if not self.enabled or self._run is None:
            return
        self._run["hparams"] = dict(params)

    def set_payload(self, key: str, payload: Dict[str, object]) -> None:
        if not self.enabled or self._run is None:
            return
        self._run[str(key)] = dict(payload)

    def track_many(self, metrics: Dict[str, object], step: int, prefix: str = "") -> None:
        if not self.enabled or self._run is None:
            return
        base = f"{prefix}/" if prefix else ""
        for key, value in metrics.items():
            if isinstance(value, bool):
                self._run.track(1.0 if value else 0.0, name=f"{base}{key}", step=int(step))
            elif isinstance(value, (int, float)):
                v = float(value)
                if math.isfinite(v):
                    self._run.track(v, name=f"{base}{key}", step=int(step))

    def track_pairs(
        self,
        pairs: Iterable[tuple[str, object]],
        step: int,
        prefix: str = "",
    ) -> None:
        self.track_many({k: v for k, v in pairs}, step=step, prefix=prefix)

    def close(self) -> None:
        if not self.enabled or self._run is None:
            return
        try:
            self._run.close()
        except Exception:
            pass
