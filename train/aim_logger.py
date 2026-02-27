"""Lightweight optional Aim metric logging helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, Iterable


class AimLogger:
    @staticmethod
    def _try_load_aim_run(repo_path: str):
        try:
            from aim import Run
            return Run
        except Exception:
            pass
        pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        fallback_site = (
            Path(repo_path).resolve()
            / ".venv"
            / "lib"
            / pyver
            / "site-packages"
        )
        if fallback_site.exists():
            sys.path.append(str(fallback_site))
            try:
                from aim import Run
                return Run
            except Exception:
                return None
        return None

    def __init__(
        self,
        enabled: bool,
        experiment: str,
        repo_path: str,
        run_name: str | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self._run = None
        if not self.enabled:
            return
        Run = self._try_load_aim_run(repo_path)
        if Run is None:
            print(
                "[aim] disabled: Python package 'aim' is not available in the training runtime",
                file=sys.stderr,
            )
            self.enabled = False
            return
        repo = str(Path(repo_path).resolve())
        Path(repo).mkdir(parents=True, exist_ok=True)
        try:
            self._run = Run(repo=repo, experiment=str(experiment))
        except Exception as exc:
            print(
                f"[aim] disabled: failed to initialize run at repo '{repo}': {exc}",
                file=sys.stderr,
            )
            self.enabled = False
            self._run = None
            return
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
