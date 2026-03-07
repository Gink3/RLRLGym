#!/usr/bin/env python3
"""PyQt6 Scenario Editor for race/class agent roster scenarios."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Dict

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlrlgym.classes import load_classes  # noqa: E402
from rlrlgym.env import EnvConfig  # noqa: E402
from rlrlgym.profiles import load_profiles  # noqa: E402
from rlrlgym.races import load_races  # noqa: E402
from rlrlgym.scenario import (  # noqa: E402
    SCENARIO_AGENTS_FILE,
    SCENARIO_ENV_FILE,
    Scenario,
    ScenarioAgent,
    agent_combined_payload,
    load_scenario,
    make_all_race_class_combinations,
    save_scenario,
)
from train.network_config import load_network_configs  # noqa: E402


class AgentDialog(QDialog):
    def __init__(
        self,
        *,
        index: int,
        races: Dict[str, object],
        classes: Dict[str, object],
        profiles: Dict[str, object],
        networks: Dict[str, object],
        existing: ScenarioAgent | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Agent Editor: agent_{index}")
        self.resize(920, 680)
        self._index = index
        self._races = races
        self._classes = classes
        self._profiles = profiles
        self._networks = networks
        self._existing = existing
        self.result_agent: ScenarioAgent | None = None

        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        layout.addLayout(top)

        top.addWidget(QLabel("Name"))
        self.name_edit = QLineEdit((existing.name if existing and existing.name else ""))
        self.name_edit.setMaximumWidth(180)
        top.addWidget(self.name_edit)

        top.addWidget(QLabel("Race"))
        self.race_combo = QComboBox()
        self.race_combo.addItems(sorted(self._races.keys()))
        top.addWidget(self.race_combo)

        top.addWidget(QLabel("Class"))
        self.class_combo = QComboBox()
        self.class_combo.addItems(sorted(self._classes.keys()))
        top.addWidget(self.class_combo)

        top.addWidget(QLabel("Profile"))
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(["<none>"] + sorted(self._profiles.keys()))
        top.addWidget(self.profile_combo)

        top.addWidget(QLabel("Network"))
        self.network_combo = QComboBox()
        self.network_combo.addItems(["<none>"] + sorted(self._networks.keys()))
        top.addWidget(self.network_combo)

        self.json_edit = QPlainTextEdit()
        self.json_edit.setFont(QFont("DejaVu Sans Mono", 10))
        layout.addWidget(self.json_edit, stretch=1)

        btn_row = QHBoxLayout()
        layout.addLayout(btn_row)
        save_btn = QPushButton("Save Agent")
        cancel_btn = QPushButton("Cancel")
        btn_row.addWidget(save_btn)
        btn_row.addWidget(cancel_btn)

        seed_race = existing.race if existing and existing.race in self._races else (sorted(self._races.keys())[0] if self._races else "")
        seed_class = existing.class_name if existing and existing.class_name in self._classes else (sorted(self._classes.keys())[0] if self._classes else "")
        self.race_combo.setCurrentText(seed_race)
        self.class_combo.setCurrentText(seed_class)
        if existing and existing.profile:
            self.profile_combo.setCurrentText(existing.profile)
        if existing and existing.network:
            self.network_combo.setCurrentText(existing.network)

        self.race_combo.currentTextChanged.connect(self._write_template)
        self.class_combo.currentTextChanged.connect(self._write_template)
        self.profile_combo.currentTextChanged.connect(self._write_template)
        self.network_combo.currentTextChanged.connect(self._write_template)
        self.name_edit.textChanged.connect(self._write_template)
        save_btn.clicked.connect(self._on_save)
        cancel_btn.clicked.connect(self.reject)

        self._write_template()

    def _write_template(self) -> None:
        race = self.race_combo.currentText().strip()
        class_name = self.class_combo.currentText().strip()
        obs = dict(self._existing.observation_config) if self._existing else {}
        payload = agent_combined_payload(
            agent_id=f"agent_{self._index}",
            race=race,
            class_name=class_name,
            name=(self.name_edit.text().strip() or None),
            profile=None if self.profile_combo.currentText() == "<none>" else self.profile_combo.currentText(),
            network=None if self.network_combo.currentText() == "<none>" else self.network_combo.currentText(),
            observation_config=obs,
            race_row=getattr(self._races.get(race), "__dict__", {}),
            class_row=getattr(self._classes.get(class_name), "__dict__", {}),
        )
        self.json_edit.setPlainText(json.dumps(payload, indent=2))

    def _on_save(self) -> None:
        try:
            payload = json.loads(self.json_edit.toPlainText().strip() or "{}")
            if not isinstance(payload, dict):
                raise ValueError("Agent payload must be a JSON object")
            race = str(payload.get("race", "")).strip()
            class_name = str(payload.get("class", payload.get("class_name", ""))).strip()
            if race not in self._races:
                raise ValueError(f"Unknown race '{race}'")
            if class_name not in self._classes:
                raise ValueError(f"Unknown class '{class_name}'")
            obs = payload.get("observation_config", {}) or {}
            if not isinstance(obs, dict):
                raise ValueError("observation_config must be an object")
            profile = payload.get("profile")
            profile = None if profile in (None, "") else str(profile)
            network = payload.get("network")
            network = None if network in (None, "") else str(network)
            name = payload.get("name")
            name = None if name in (None, "") else str(name)
            self.result_agent = ScenarioAgent(
                agent_id=f"agent_{self._index}",
                race=race,
                class_name=class_name,
                name=name,
                profile=profile,
                network=network,
                observation_config=dict(obs),
            )
            self.accept()
        except Exception as exc:
            QMessageBox.critical(self, "Invalid agent", str(exc))


class ScenarioEditorWindow(QMainWindow):
    def __init__(self, initial_path: Path | None = None) -> None:
        super().__init__()
        self.setWindowTitle("RLRLGym Scenario Editor (PyQt6)")
        self.resize(1300, 850)

        self.races = load_races("data/base/agent/agent_races.json")
        self.classes = load_classes("data/base/agent/agent_classes.json")
        self.profiles = load_profiles("data/base/agent/agent_profiles.json")
        self.networks = load_network_configs("data/base/agent/agent_networks.json")
        self.default_env_config = self._load_default_env_config()
        self.maps_dir = Path("data/maps")

        self.scenario_path: Path | None = initial_path
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
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        toolbar = self.menuBar().addMenu("File")
        new_act = QAction("New", self)
        load_act = QAction("Load", self)
        save_act = QAction("Save", self)
        save_as_act = QAction("Save As", self)
        toolbar.addAction(new_act)
        toolbar.addAction(load_act)
        toolbar.addAction(save_act)
        toolbar.addAction(save_as_act)
        new_act.triggered.connect(self._new)
        load_act.triggered.connect(self._load_dialog)
        save_act.triggered.connect(self._save)
        save_as_act.triggered.connect(self._save_as)

        top = QHBoxLayout()
        layout.addLayout(top)

        top.addWidget(QLabel("Scenario Name"))
        self.name_edit = QLineEdit(self.scenario.name)
        self.name_edit.setMaximumWidth(420)
        top.addWidget(self.name_edit)

        self.generate_btn = QPushButton("Generate Race x Class")
        self.generate_btn.clicked.connect(self._generate_combos)
        top.addWidget(self.generate_btn)
        top.addWidget(QLabel("Static Map"))
        self.map_combo = QComboBox()
        self.map_combo.setMinimumWidth(280)
        top.addWidget(self.map_combo)
        self.refresh_maps_btn = QPushButton("Refresh Maps")
        top.addWidget(self.refresh_maps_btn)
        top.addStretch(1)

        body = QHBoxLayout()
        layout.addLayout(body, stretch=1)

        left = QVBoxLayout()
        body.addLayout(left, stretch=2)

        self.agent_list = QListWidget()
        left.addWidget(self.agent_list, stretch=1)

        left_btns = QHBoxLayout()
        left.addLayout(left_btns)
        self.add_agent_btn = QPushButton("Add Agent")
        self.edit_agent_btn = QPushButton("Edit Agent")
        self.remove_agent_btn = QPushButton("Remove Agent")
        left_btns.addWidget(self.add_agent_btn)
        left_btns.addWidget(self.edit_agent_btn)
        left_btns.addWidget(self.remove_agent_btn)
        self.add_agent_btn.clicked.connect(self._add_agent)
        self.edit_agent_btn.clicked.connect(self._edit_selected)
        self.remove_agent_btn.clicked.connect(self._remove_selected)

        right = QVBoxLayout()
        body.addLayout(right, stretch=3)
        right.addWidget(QLabel("Scenario env_config (full editable config)"))
        self.env_edit = QPlainTextEdit()
        self.env_edit.setFont(QFont("DejaVu Sans Mono", 10))
        right.addWidget(self.env_edit, stretch=1)
        self.env_edit.setPlainText(json.dumps(self.scenario.env_config, indent=2))
        self.refresh_maps_btn.clicked.connect(self._refresh_map_choices)
        self.map_combo.currentTextChanged.connect(self._apply_selected_map)
        self._refresh_map_choices()
        self._sync_map_selector_from_env()

    def _load_default_env_config(self) -> Dict[str, object]:
        try:
            cfg = EnvConfig.from_json("data/env_config.json")
        except Exception:
            cfg = EnvConfig()
        out = dict(vars(cfg))
        out.pop("agent_scenario", None)
        return self._expand_embedded_spawn_data(out)

    def _read_json_file(self, path_value: object, fallback: str) -> Dict[str, object]:
        p = Path(str(path_value or fallback))
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Expected JSON object at {p}")
        return raw

    def _expand_embedded_spawn_data(self, env_cfg: Dict[str, object]) -> Dict[str, object]:
        out = dict(env_cfg)
        if not out.get("structures_data"):
            out["structures_data"] = self._read_json_file(out.get("tiles_path"), "data/base/tiles.json")
        if not out.get("items_data"):
            out["items_data"] = self._read_json_file(out.get("items_path"), "data/base/items.json")
        if not out.get("monsters_data"):
            out["monsters_data"] = self._read_json_file(out.get("monsters_path"), "data/base/monsters.json")
        if not out.get("animals_data"):
            out["animals_data"] = self._read_json_file(out.get("animals_path"), "data/base/animals.json")
        if not out.get("monster_spawns_data"):
            out["monster_spawns_data"] = self._read_json_file(out.get("monster_spawns_path"), "data/base/monster_spawns.json")
        if not out.get("mapgen_config_data"):
            out["mapgen_config_data"] = self._read_json_file(out.get("mapgen_config_path"), "data/base/mapgen_config.json")
        if not out.get("map_structures_data"):
            out["map_structures_data"] = self._read_json_file(out.get("map_structures_path"), "data/base/structures.json")
        if not out.get("recipes_data"):
            out["recipes_data"] = self._read_json_file(out.get("recipes_path"), "data/base/recipes.json")
        if not out.get("statuses_data"):
            out["statuses_data"] = self._read_json_file(out.get("statuses_path"), "data/base/statuses.json")
        if not out.get("spells_data"):
            out["spells_data"] = self._read_json_file(out.get("spells_path"), "data/base/spells.json")
        if not out.get("enchantments_data"):
            out["enchantments_data"] = self._read_json_file(out.get("enchantments_path"), "data/base/enchantments.json")
        return out

    def _new(self) -> None:
        self.scenario_path = None
        self.scenario = Scenario(name="new_scenario", env_config=copy.deepcopy(self.default_env_config), agents=[])
        self.name_edit.setText(self.scenario.name)
        self.env_edit.setPlainText(json.dumps(self.scenario.env_config, indent=2))
        self._refresh_map_choices()
        self._sync_map_selector_from_env()
        self._refresh_list()

    def _load_dialog(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Load scenario directory", "data/scenarios")
        if not chosen:
            return
        self._load(Path(chosen))

    def _load(self, path: Path) -> None:
        try:
            self.scenario = load_scenario(path)
            self.scenario_path = path
            merged_env = copy.deepcopy(self.default_env_config)
            for k, v in dict(self.scenario.env_config).items():
                merged_env[k] = v
            self.scenario.env_config = self._expand_embedded_spawn_data(merged_env)
            self.name_edit.setText(self.scenario.name)
            self.env_edit.setPlainText(json.dumps(self.scenario.env_config, indent=2))
            self._refresh_map_choices()
            self._sync_map_selector_from_env()
            self._refresh_list()
        except Exception as exc:
            QMessageBox.critical(self, "Load failed", str(exc))

    def _save_as(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Select scenario save directory", "data/scenarios")
        if not chosen:
            return
        name = self.name_edit.text().strip() or "scenario"
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name)
        self.scenario_path = Path(chosen) / safe
        self._save()

    def _save(self) -> None:
        if self.scenario_path is None:
            self._save_as()
            return
        try:
            env_payload = json.loads(self.env_edit.toPlainText().strip() or "{}")
            if not isinstance(env_payload, dict):
                raise ValueError("env_config must be a JSON object")
            clean_env: Dict[str, object] = {}
            for key in self.default_env_config.keys():
                clean_env[key] = env_payload[key] if key in env_payload else self.default_env_config[key]
            clean_env = self._expand_embedded_spawn_data(clean_env)
            self.scenario.name = self.name_edit.text().strip() or "scenario"
            self.scenario.env_config = clean_env
            out_dir = save_scenario(self.scenario_path, self.scenario)
            self.scenario_path = out_dir
            QMessageBox.information(
                self,
                "Saved",
                (
                    f"Saved scenario directory: {out_dir}\n"
                    f"- {out_dir / SCENARIO_ENV_FILE}\n"
                    f"- {out_dir / SCENARIO_AGENTS_FILE}"
                ),
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))

    def _refresh_list(self) -> None:
        self.agent_list.clear()
        for a in self.scenario.agents:
            display_name = (a.name or "").strip() or f"{a.race}/{a.class_name}"
            self.agent_list.addItem(display_name)

    def _refresh_map_choices(self) -> None:
        self.map_combo.blockSignals(True)
        self.map_combo.clear()
        self.map_combo.addItem("<none>")
        if self.maps_dir.exists():
            for path in sorted(self.maps_dir.glob("*.json")):
                self.map_combo.addItem(path.as_posix())
        self.map_combo.blockSignals(False)

    def _sync_map_selector_from_env(self) -> None:
        try:
            env_payload = json.loads(self.env_edit.toPlainText().strip() or "{}")
        except Exception:
            return
        if not isinstance(env_payload, dict):
            return
        map_path = str(env_payload.get("static_map_path", "")).strip()
        idx = self.map_combo.findText(map_path)
        if idx < 0:
            idx = 0
        self.map_combo.blockSignals(True)
        self.map_combo.setCurrentIndex(idx)
        self.map_combo.blockSignals(False)

    def _apply_selected_map(self, selected: str) -> None:
        try:
            env_payload = json.loads(self.env_edit.toPlainText().strip() or "{}")
            if not isinstance(env_payload, dict):
                return
            if selected and selected != "<none>":
                env_payload["static_map_path"] = selected
                env_payload["static_map_data"] = {}
            else:
                env_payload["static_map_path"] = ""
            self.env_edit.setPlainText(json.dumps(env_payload, indent=2))
        except Exception:
            return

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
        return int(self.agent_list.currentRow())

    def _add_agent(self) -> None:
        idx = len(self.scenario.agents)
        dlg = AgentDialog(
            index=idx,
            races=self.races,
            classes=self.classes,
            profiles=self.profiles,
            networks=self.networks,
            existing=None,
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.result_agent is not None:
            self.scenario.agents.append(dlg.result_agent)
            self._refresh_list()

    def _edit_selected(self) -> None:
        idx = self._selected_index()
        if idx < 0 or idx >= len(self.scenario.agents):
            return
        dlg = AgentDialog(
            index=idx,
            races=self.races,
            classes=self.classes,
            profiles=self.profiles,
            networks=self.networks,
            existing=self.scenario.agents[idx],
            parent=self,
        )
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.result_agent is not None:
            self.scenario.agents[idx] = dlg.result_agent
            self._refresh_list()

    def _remove_selected(self) -> None:
        idx = self._selected_index()
        if idx < 0 or idx >= len(self.scenario.agents):
            return
        del self.scenario.agents[idx]
        for i, agent in enumerate(self.scenario.agents):
            agent.agent_id = f"agent_{i}"
        self._refresh_list()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Open Scenario Editor (PyQt6)")
    p.add_argument("--scenario", type=str, default="", help="Optional scenario directory path")
    return p


def main() -> None:
    args = build_parser().parse_args()
    initial = Path(args.scenario).resolve() if args.scenario else None
    app = QApplication(sys.argv)
    win = ScenarioEditorWindow(initial_path=initial)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
