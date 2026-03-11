"""Helpers for suppressing known noisy RLlib transitional warnings."""

from __future__ import annotations

import logging
import warnings


_LEGACY_RLMODULE_CONFIG_SNIPPET = "`RLModule(config=[RLModuleConfig object])` has been deprecated."


class _MessageFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return True
        return _LEGACY_RLMODULE_CONFIG_SNIPPET not in str(msg)


def install_rllib_warning_filters() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*RLModule\(config=\[RLModuleConfig object\]\) has been deprecated.*",
        category=DeprecationWarning,
    )
    for name in (
        "ray.rllib",
        "ray.rllib.core.rl_module.rl_module",
        "ray.rllib.utils.deprecation",
        "ray._common.deprecation",
    ):
        logger = logging.getLogger(name)
        if not any(isinstance(flt, _MessageFilter) for flt in logger.filters):
            logger.addFilter(_MessageFilter())
