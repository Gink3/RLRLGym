"""Shared UI theme definitions and persistence for local tools."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

THEMES: Dict[str, Dict[str, str]] = {
    "storm_violet": {
        "bg": "#111827",
        "panel": "#1F2937",
        "panel_alt": "#374151",
        "text": "#F3F4F6",
        "text_muted": "#9CA3AF",
        "primary": "#9333EA",
        "primary_hover": "#7E22CE",
        "accent": "#C084FC",
        "border": "#4B5563",
        "danger": "#F87171",
        "success": "#34D399",
    },
    "muted_slate": {
        "bg": "#0F172A",
        "panel": "#1E293B",
        "panel_alt": "#334155",
        "text": "#E2E8F0",
        "text_muted": "#94A3B8",
        "primary": "#8B5CF6",
        "primary_hover": "#7C3AED",
        "accent": "#A78BFA",
        "border": "#475569",
        "danger": "#EF4444",
        "success": "#22C55E",
    },
    "graphite_orchid": {
        "bg": "#121418",
        "panel": "#1F242D",
        "panel_alt": "#2B3340",
        "text": "#E6EAF2",
        "text_muted": "#9AA5B1",
        "primary": "#9D4EDD",
        "primary_hover": "#7B2CBF",
        "accent": "#C77DFF",
        "border": "#3A4455",
        "danger": "#F43F5E",
        "success": "#10B981",
    },
}

DEFAULT_THEME_NAME = "storm_violet"
SETTINGS_PATH = Path("data") / "user" / "tool_settings.json"


def list_theme_names() -> list[str]:
    return sorted(THEMES.keys())


def theme_label(name: str) -> str:
    return str(name).replace("_", " ").title()


def get_theme(name: str) -> Dict[str, str]:
    if name in THEMES:
        return dict(THEMES[name])
    return dict(THEMES[DEFAULT_THEME_NAME])


def _load_settings() -> Dict[str, object]:
    try:
        raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            return raw
    except Exception:
        pass
    return {}


def _save_settings(payload: Dict[str, object]) -> None:
    SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_selected_theme() -> str:
    settings = _load_settings()
    name = str(settings.get("theme", DEFAULT_THEME_NAME)).strip()
    return name if name in THEMES else DEFAULT_THEME_NAME


def save_selected_theme(name: str) -> str:
    use_name = name if name in THEMES else DEFAULT_THEME_NAME
    settings = _load_settings()
    settings["theme"] = use_name
    _save_settings(settings)
    return use_name
