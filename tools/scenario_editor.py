"""Tkinter Scenario Editor for race/class agent roster scenarios."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Dict, List

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Tkinter is required for Scenario Editor") from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlrlgym.classes import load_classes
from rlrlgym.env import EnvConfig
from rlrlgym.profiles import load_profiles
from rlrlgym.races import load_races
from rlrlgym.scenario import (
    Scenario,
    ScenarioAgent,
    agent_combined_payload,
    load_scenario,
    make_all_race_class_combinations,
    save_scenario,
)
from rlrlgym.themes import (
    get_theme,
    list_theme_names,
    load_selected_theme,
    save_selected_theme,
    theme_label,
)
from train.network_config import load_network_configs


class ScenarioEditorApp:
    def __init__(self, root: tk.Tk, initial_path: Path | None = None) -> None:
        self.root = root
        self.root.title("RLRLGym Scenario Editor")

        self.races = load_races("data/base/agent_races.json")
        self.classes = load_classes("data/base/agent_classes.json")
        self.profiles = load_profiles("data/base/agent_profiles.json")
        self.networks = load_network_configs("data/base/agent_networks.json")
        self.default_env_config = self._load_default_env_config()
        self._theme_name = load_selected_theme()
        self._theme = get_theme(self._theme_name)

        self.scenario_path = initial_path
        self.scenario = Scenario(
            name="new_scenario",
            env_config=copy.deepcopy(self.default_env_config),
            agents=[],
        )

        self._build_ui()
        if initial_path and initial_path.exists():
            self._load(initial_path)
        else:
            self._refresh_list()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=8)

        ttk.Button(top, text="New", command=self._new).pack(side="left")
        ttk.Button(top, text="Load", command=self._load_dialog).pack(side="left", padx=(4, 0))
        ttk.Button(top, text="Save", command=self._save).pack(side="left", padx=(4, 0))
        ttk.Button(top, text="Save As", command=self._save_as).pack(side="left", padx=(4, 0))
        ttk.Button(top, text="Generate Race x Class", command=self._generate_combos).pack(
            side="left", padx=(10, 0)
        )
        self.theme_var = tk.StringVar(value=self._theme_name)
        self.settings_button = ttk.Menubutton(top, text="Settings")
        self.settings_menu = tk.Menu(self.settings_button, tearoff=0)
        self.theme_menu = tk.Menu(self.settings_menu, tearoff=0)
        for name in list_theme_names():
            self.theme_menu.add_radiobutton(
                label=theme_label(name),
                value=name,
                variable=self.theme_var,
                command=self._on_theme_change,
            )
        self.settings_menu.add_cascade(label="Theme", menu=self.theme_menu)
        self.settings_button["menu"] = self.settings_menu
        self.settings_button.pack(side="right")

        name_row = ttk.Frame(self.root)
        name_row.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(name_row, text="Scenario Name").pack(side="left")
        self.name_var = tk.StringVar(value=self.scenario.name)
        ttk.Entry(name_row, textvariable=self.name_var, width=40).pack(side="left", padx=(8, 0))

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        left = ttk.Frame(body)
        left.pack(side="left", fill="both", expand=True)

        self.agent_list = tk.Listbox(left, height=18)
        self.agent_list.pack(fill="both", expand=True)

        btns = ttk.Frame(left)
        btns.pack(fill="x", pady=(6, 0))
        ttk.Button(btns, text="Add Agent", command=self._add_agent).pack(side="left")
        ttk.Button(btns, text="Edit Agent", command=self._edit_selected).pack(side="left", padx=(4, 0))
        ttk.Button(btns, text="Remove Agent", command=self._remove_selected).pack(
            side="left", padx=(4, 0)
        )

        right = ttk.Frame(body)
        right.pack(side="left", fill="both", expand=True, padx=(8, 0))
        ttk.Label(right, text="Scenario env_config (full editable config)").pack(anchor="w")
        self.env_text = tk.Text(right, width=64, height=12, font=("Courier", 10))
        self.env_text.pack(fill="both", expand=True)
        self.env_text.insert("1.0", json.dumps(self.scenario.env_config, indent=2))
        self._apply_theme()

    def _load_default_env_config(self) -> Dict[str, object]:
        try:
            cfg = EnvConfig.from_json("data/env_config.json")
        except Exception:
            cfg = EnvConfig()
        out = dict(vars(cfg))
        out.pop("agent_scenario", None)
        return out

    def _on_theme_change(self) -> None:
        name = save_selected_theme(self.theme_var.get())
        self.theme_var.set(name)
        self._theme_name = name
        self._theme = get_theme(name)
        self._apply_theme()

    def _apply_theme(self) -> None:
        t = self._theme
        self.root.configure(bg=t["bg"])
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure(".", background=t["panel"], foreground=t["text"])
        style.configure("TFrame", background=t["panel"])
        style.configure("TLabel", background=t["panel"], foreground=t["text"])
        style.configure("TButton", background=t["panel_alt"], foreground=t["text"])
        style.map("TButton", background=[("active", t["primary_hover"])])
        style.configure("TMenubutton", background=t["panel_alt"], foreground=t["text"])
        style.configure(
            "TCombobox",
            fieldbackground=t["panel_alt"],
            background=t["panel_alt"],
            foreground=t["text"],
        )
        self.settings_menu.configure(
            background=t["panel"],
            foreground=t["text"],
            activebackground=t["primary"],
            activeforeground=t["text"],
            tearoff=False,
        )
        self.theme_menu.configure(
            background=t["panel"],
            foreground=t["text"],
            activebackground=t["primary"],
            activeforeground=t["text"],
            tearoff=False,
        )
        self.agent_list.configure(
            bg=t["bg"],
            fg=t["text"],
            selectbackground=t["primary"],
            selectforeground=t["text"],
            highlightbackground=t["border"],
            highlightcolor=t["primary"],
        )
        self.env_text.configure(bg=t["bg"], fg=t["text"], insertbackground=t["text"])

    def _new(self) -> None:
        self.scenario_path = None
        self.scenario = Scenario(
            name="new_scenario",
            env_config=copy.deepcopy(self.default_env_config),
            agents=[],
        )
        self.name_var.set(self.scenario.name)
        self.env_text.delete("1.0", tk.END)
        self.env_text.insert("1.0", json.dumps(self.scenario.env_config, indent=2))
        self._refresh_list()

    def _load_dialog(self) -> None:
        chosen = filedialog.askopenfilename(
            title="Load scenario",
            filetypes=[("JSON", "*.json"), ("All Files", "*.*")],
            initialdir="data",
        )
        if not chosen:
            return
        self._load(Path(chosen))

    def _load(self, path: Path) -> None:
        try:
            self.scenario = load_scenario(path)
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        self.scenario_path = path
        self.name_var.set(self.scenario.name)
        merged_env = copy.deepcopy(self.default_env_config)
        for key, value in dict(self.scenario.env_config).items():
            merged_env[key] = value
        self.scenario.env_config = merged_env
        self.env_text.delete("1.0", tk.END)
        self.env_text.insert("1.0", json.dumps(self.scenario.env_config, indent=2))
        self._refresh_list()

    def _save_as(self) -> None:
        chosen = filedialog.asksaveasfilename(
            title="Save scenario",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
            initialdir="data",
        )
        if not chosen:
            return
        self.scenario_path = Path(chosen)
        self._save()

    def _save(self) -> None:
        if self.scenario_path is None:
            self._save_as()
            return
        try:
            env_payload = json.loads(self.env_text.get("1.0", tk.END).strip() or "{}")
            if not isinstance(env_payload, dict):
                raise ValueError("env_config must be a JSON object")
            clean_env: Dict[str, object] = {}
            for key in self.default_env_config.keys():
                if key in env_payload:
                    clean_env[key] = env_payload[key]
                else:
                    clean_env[key] = self.default_env_config[key]
            self.scenario.name = self.name_var.get().strip() or "scenario"
            self.scenario.env_config = clean_env
            save_scenario(self.scenario_path, self.scenario)
            messagebox.showinfo("Saved", f"Saved to {self.scenario_path}")
        except Exception as exc:
            messagebox.showerror("Save failed", str(exc))

    def _refresh_list(self) -> None:
        self.agent_list.delete(0, tk.END)
        for i, a in enumerate(self.scenario.agents):
            self.agent_list.insert(
                tk.END,
                f"{i:02d}  {a.agent_id}  race={a.race} class={a.class_name} profile={a.profile or '-'} network={a.network or '-'}",
            )

    def _generate_combos(self) -> None:
        default_profile_by_race = {name: name for name in self.races.keys() if name in self.profiles}
        default_network = "default" if "default" in self.networks else None
        self.scenario.agents = make_all_race_class_combinations(
            list(self.races.keys()),
            list(self.classes.keys()),
            default_profile_by_race=default_profile_by_race,
            default_network=default_network,
        )
        self._refresh_list()

    def _selected_index(self) -> int:
        selected = list(self.agent_list.curselection())
        if not selected:
            return -1
        return int(selected[0])

    def _add_agent(self) -> None:
        idx = len(self.scenario.agents)
        agent = self._open_agent_editor(index=idx, existing=None)
        if agent is None:
            return
        self.scenario.agents.append(agent)
        self._refresh_list()

    def _edit_selected(self) -> None:
        idx = self._selected_index()
        if idx < 0:
            return
        existing = self.scenario.agents[idx]
        edited = self._open_agent_editor(index=idx, existing=existing)
        if edited is None:
            return
        self.scenario.agents[idx] = edited
        self._refresh_list()

    def _remove_selected(self) -> None:
        idx = self._selected_index()
        if idx < 0:
            return
        del self.scenario.agents[idx]
        for i, agent in enumerate(self.scenario.agents):
            agent.agent_id = f"agent_{i}"
        self._refresh_list()

    def _open_agent_editor(self, index: int, existing: ScenarioAgent | None) -> ScenarioAgent | None:
        dialog = tk.Toplevel(self.root)
        dialog.title(f"Agent Editor: agent_{index}")
        dialog.transient(self.root)
        dialog.grab_set()

        race_names = sorted(self.races.keys())
        class_names = sorted(self.classes.keys())
        profile_names = ["<none>"] + sorted(self.profiles.keys())
        network_names = ["<none>"] + sorted(self.networks.keys())

        seed_race = existing.race if existing else (race_names[0] if race_names else "")
        seed_class = existing.class_name if existing else (class_names[0] if class_names else "")
        seed_profile = existing.profile if existing and existing.profile else "<none>"
        seed_network = existing.network if existing and existing.network else "<none>"

        row = ttk.Frame(dialog)
        row.pack(fill="x", padx=8, pady=8)
        ttk.Label(row, text="Race").pack(side="left")
        race_var = tk.StringVar(value=seed_race)
        race_box = ttk.Combobox(row, textvariable=race_var, values=race_names, width=16, state="readonly")
        race_box.pack(side="left", padx=(6, 12))
        ttk.Label(row, text="Class").pack(side="left")
        class_var = tk.StringVar(value=seed_class)
        class_box = ttk.Combobox(row, textvariable=class_var, values=class_names, width=16, state="readonly")
        class_box.pack(side="left", padx=(6, 12))
        ttk.Label(row, text="Profile").pack(side="left")
        profile_var = tk.StringVar(value=seed_profile)
        profile_box = ttk.Combobox(
            row,
            textvariable=profile_var,
            values=profile_names,
            width=14,
            state="readonly",
        )
        profile_box.pack(side="left", padx=(6, 12))
        ttk.Label(row, text="Network").pack(side="left")
        network_var = tk.StringVar(value=seed_network)
        network_box = ttk.Combobox(
            row,
            textvariable=network_var,
            values=network_names,
            width=14,
            state="readonly",
        )
        network_box.pack(side="left", padx=(6, 0))

        ttk.Label(dialog, text="Combined JSON (final manual edit)").pack(anchor="w", padx=8)
        editor = tk.Text(dialog, width=96, height=22, font=("Courier", 10))
        editor.pack(fill="both", expand=True, padx=8, pady=(2, 8))

        existing_obs = dict(existing.observation_config) if existing else {}

        def write_template(*_args):
            race = race_var.get().strip()
            class_name = class_var.get().strip()
            payload = agent_combined_payload(
                agent_id=f"agent_{index}",
                race=race,
                class_name=class_name,
                profile=None if profile_var.get() == "<none>" else profile_var.get(),
                network=None if network_var.get() == "<none>" else network_var.get(),
                observation_config=existing_obs,
                race_row=self.races[race].__dict__ if race in self.races else {},
                class_row=self.classes[class_name].__dict__ if class_name in self.classes else {},
            )
            editor.delete("1.0", tk.END)
            editor.insert("1.0", json.dumps(payload, indent=2))

        race_box.bind("<<ComboboxSelected>>", write_template)
        class_box.bind("<<ComboboxSelected>>", write_template)
        profile_box.bind("<<ComboboxSelected>>", write_template)
        network_box.bind("<<ComboboxSelected>>", write_template)
        write_template()

        out: Dict[str, ScenarioAgent] = {}

        def on_save() -> None:
            try:
                payload = json.loads(editor.get("1.0", tk.END).strip())
                if not isinstance(payload, dict):
                    raise ValueError("Agent payload must be a JSON object")
                race = str(payload.get("race", "")).strip()
                class_name = str(payload.get("class", payload.get("class_name", "")).strip())
                if race not in self.races:
                    raise ValueError(f"Unknown race '{race}'")
                if class_name not in self.classes:
                    raise ValueError(f"Unknown class '{class_name}'")
                obs = payload.get("observation_config", {})
                if obs is None:
                    obs = {}
                if not isinstance(obs, dict):
                    raise ValueError("observation_config must be an object")
                profile = payload.get("profile")
                if profile in ("", None):
                    profile = None
                network = payload.get("network")
                if network in ("", None):
                    network = None
                out["agent"] = ScenarioAgent(
                    agent_id=f"agent_{index}",
                    race=race,
                    class_name=class_name,
                    profile=(str(profile) if profile is not None else None),
                    network=(str(network) if network is not None else None),
                    observation_config=dict(obs),
                )
                dialog.destroy()
            except Exception as exc:
                messagebox.showerror("Invalid agent", str(exc), parent=dialog)

        btn_row = ttk.Frame(dialog)
        btn_row.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Button(btn_row, text="Save Agent", command=on_save).pack(side="left")
        ttk.Button(btn_row, text="Cancel", command=dialog.destroy).pack(side="left", padx=(4, 0))

        self.root.wait_window(dialog)
        return out.get("agent")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Open Scenario Editor")
    p.add_argument("--scenario", type=str, default="", help="Optional scenario JSON path")
    return p


def main() -> None:
    args = build_parser().parse_args()
    initial = Path(args.scenario).resolve() if args.scenario else None
    root = tk.Tk()
    app = ScenarioEditorApp(root, initial_path=initial)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()
