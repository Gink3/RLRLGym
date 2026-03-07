#!/usr/bin/env python3
"""PyQt6 launcher for training with live metric updates."""

from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
from pathlib import Path

from PyQt6.QtCore import QProcess, Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlrlgym.world.themes import (  # noqa: E402
    get_theme,
    list_theme_names,
    load_selected_theme,
    save_selected_theme,
    theme_label,
)

CUSTOM_PROGRESS_RE = re.compile(
    r"ret\d+=(?P<ret>-?\d+\.\d+)\s+"
    r"win\d+=(?P<win>-?\d+\.\d+)\s+"
    r"surv\d+=(?P<surv>-?\d+\.\d+)\s+"
    r"starve\d+=(?P<starve>-?\d+\.\d+)\s+"
    r"loss\d+=(?P<loss>-?\d+\.\d+)\s+"
    r"eps=(?P<eps>-?\d+\.\d+)"
)


class TrainLauncherWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RLRLGym Training Launcher (PyQt6)")
        self.resize(1200, 800)

        self._theme_name = load_selected_theme()
        self._theme = get_theme(self._theme_name)

        self.proc: QProcess | None = None
        self._stdout_buf = ""

        self._build_ui()
        self._apply_theme()

    def _build_ui(self) -> None:
        root = QWidget(self)
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        form = QFormLayout()
        layout.addLayout(form)

        line1 = QHBoxLayout()
        self.backend_combo = QComboBox()
        self.backend_combo.addItems(["custom", "rllib"])
        self.seed_edit = QLineEdit("0")
        self.seed_edit.setMaximumWidth(110)
        self.episodes_edit = QLineEdit("100")
        self.episodes_edit.setMaximumWidth(110)
        self.iterations_edit = QLineEdit("50")
        self.iterations_edit.setMaximumWidth(110)
        self.max_steps_edit = QLineEdit("120")
        self.max_steps_edit.setMaximumWidth(110)

        line1.addWidget(QLabel("Backend"))
        line1.addWidget(self.backend_combo)
        line1.addSpacing(10)
        line1.addWidget(QLabel("Seed"))
        line1.addWidget(self.seed_edit)
        line1.addSpacing(10)
        line1.addWidget(QLabel("Episodes"))
        line1.addWidget(self.episodes_edit)
        line1.addSpacing(10)
        line1.addWidget(QLabel("Iterations"))
        line1.addWidget(self.iterations_edit)
        line1.addSpacing(10)
        line1.addWidget(QLabel("Max Steps"))
        line1.addWidget(self.max_steps_edit)
        line1.addStretch(1)
        form.addRow(line1)

        out_row = QHBoxLayout()
        self.output_edit = QLineEdit("outputs/train/gui_run")
        self.output_btn = QPushButton("Browse")
        out_row.addWidget(QLabel("Output Dir"))
        out_row.addWidget(self.output_edit, stretch=1)
        out_row.addWidget(self.output_btn)
        form.addRow(out_row)

        scenario_row = QHBoxLayout()
        self.scenario_edit = QLineEdit("")
        self.scenario_btn = QPushButton("Browse")
        scenario_row.addWidget(QLabel("Scenario Path"))
        scenario_row.addWidget(self.scenario_edit, stretch=1)
        scenario_row.addWidget(self.scenario_btn)
        form.addRow(scenario_row)

        controls = QHBoxLayout()
        self.start_btn = QPushButton("Start Training")
        self.stop_btn = QPushButton("Stop")
        controls.addWidget(self.start_btn)
        controls.addWidget(self.stop_btn)
        controls.addStretch(1)

        self.settings_btn = QToolButton()
        self.settings_btn.setText("Settings")
        self.settings_btn.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.settings_menu = QMenu(self)
        self.theme_menu = self.settings_menu.addMenu("Theme")
        self.theme_actions: dict[str, QAction] = {}
        for name in list_theme_names():
            act = QAction(theme_label(name), self)
            act.setCheckable(True)
            act.triggered.connect(lambda checked, n=name: self._on_theme_change(n, checked))
            self.theme_menu.addAction(act)
            self.theme_actions[name] = act
        self.settings_btn.setMenu(self.settings_menu)
        controls.addWidget(self.settings_btn)
        layout.addLayout(controls)

        metrics_box = QWidget(self)
        metrics_layout = QGridLayout(metrics_box)
        layout.addWidget(metrics_box)

        self.metric_labels: dict[str, QLabel] = {}
        metric_order = [
            ("ret", "Return"),
            ("win", "Win"),
            ("surv", "Survival"),
            ("starve", "Starve"),
            ("loss", "Loss"),
            ("eps", "Epsilon"),
            ("status", "Status"),
        ]
        for col, (key, label) in enumerate(metric_order):
            metrics_layout.addWidget(QLabel(label), 0, col)
            value = QLabel("idle" if key == "status" else "-")
            self.metric_labels[key] = value
            metrics_layout.addWidget(value, 1, col)

        self.log_text = QTextEdit(self)
        self.log_text.setReadOnly(True)
        self.log_text.setLineWrapMode(QTextEdit.LineWrapMode.NoWrap)
        layout.addWidget(self.log_text, stretch=1)

        self.output_btn.clicked.connect(self._pick_output_dir)
        self.scenario_btn.clicked.connect(self._pick_scenario)
        self.start_btn.clicked.connect(self._start)
        self.stop_btn.clicked.connect(self._stop)
        self.backend_combo.currentTextChanged.connect(self._on_backend_changed)
        self._on_backend_changed(self.backend_combo.currentText())

        current = self._theme_name if self._theme_name in self.theme_actions else next(iter(self.theme_actions.keys()), "")
        if current:
            self.theme_actions[current].setChecked(True)

    def _on_backend_changed(self, backend: str) -> None:
        is_custom = backend == "custom"
        self.episodes_edit.setEnabled(is_custom)
        self.iterations_edit.setEnabled(not is_custom)

    def _on_theme_change(self, name: str, checked: bool) -> None:
        if not checked:
            return
        saved = save_selected_theme(name)
        self._theme_name = saved
        self._theme = get_theme(saved)
        for key, action in self.theme_actions.items():
            action.setChecked(key == saved)
        self._apply_theme()

    def _apply_theme(self) -> None:
        t = self._theme
        style = f"""
            QMainWindow {{ background: {t['bg']}; color: {t['text']}; }}
            QWidget {{ background: {t['panel']}; color: {t['text']}; }}
            QLineEdit, QComboBox, QTextEdit {{
                background: {t['panel_alt']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 4px;
                padding: 4px;
            }}
            QPushButton, QToolButton {{
                background: {t['panel_alt']};
                color: {t['text']};
                border: 1px solid {t['border']};
                border-radius: 4px;
                padding: 5px 9px;
            }}
            QPushButton:hover, QToolButton:hover {{
                background: {t['primary_hover']};
            }}
            QMenu {{
                background: {t['panel']};
                color: {t['text']};
                border: 1px solid {t['border']};
            }}
            QMenu::item:selected {{
                background: {t['primary']};
            }}
        """
        self.setStyleSheet(style)

    def _pick_output_dir(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Select output directory", "outputs")
        if p:
            self.output_edit.setText(str(Path(p)))

    def _pick_scenario(self) -> None:
        p = QFileDialog.getExistingDirectory(self, "Select scenario directory", "data/scenarios")
        if p:
            self.scenario_edit.setText(str(Path(p)))

    def _set_metric(self, key: str, value: str) -> None:
        label = self.metric_labels.get(key)
        if label is not None:
            label.setText(value)

    def _append_log(self, line: str) -> None:
        self.log_text.append(line)
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())

    def _parse_metrics(self, line: str) -> None:
        m = CUSTOM_PROGRESS_RE.search(line)
        if not m:
            return
        for key in ("ret", "win", "surv", "starve", "loss", "eps"):
            self._set_metric(key, m.group(key))

    def _build_command(self) -> list[str]:
        backend = self.backend_combo.currentText().strip() or "custom"
        cmd = ["python3", "-m", "train", "--backend", backend]
        cmd += ["--seed", self.seed_edit.text().strip() or "0"]
        cmd += ["--output-dir", self.output_edit.text().strip() or "outputs/train/gui_run"]

        max_steps = self.max_steps_edit.text().strip()
        if max_steps:
            cmd += ["--max-steps", max_steps]

        scenario = self.scenario_edit.text().strip()
        if scenario:
            cmd += ["--scenario-path", scenario]

        if backend == "custom":
            cmd += ["--episodes", self.episodes_edit.text().strip() or "100"]
        else:
            cmd += ["--iterations", self.iterations_edit.text().strip() or "50"]
        return cmd

    def _start(self) -> None:
        if self.proc is not None and self.proc.state() != QProcess.ProcessState.NotRunning:
            QMessageBox.warning(self, "Training", "Training is already running.")
            return

        cmd = self._build_command()
        self.proc = QProcess(self)
        self.proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

        env = self.proc.processEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        for key, value in os.environ.items():
            if not env.contains(key):
                env.insert(key, value)
        self.proc.setProcessEnvironment(env)
        self.proc.setWorkingDirectory(str(Path(__file__).resolve().parents[1]))
        self.proc.readyReadStandardOutput.connect(self._read_process_output)
        self.proc.finished.connect(self._on_process_finished)
        self.proc.errorOccurred.connect(self._on_process_error)

        self._set_metric("status", "running")
        self.start_btn.setEnabled(False)
        self._append_log("$ " + " ".join(shlex.quote(part) for part in cmd))

        self._stdout_buf = ""
        self.proc.start(cmd[0], cmd[1:])
        if not self.proc.waitForStarted(1500):
            err = self.proc.errorString() if self.proc is not None else "unknown"
            QMessageBox.critical(self, "Failed to start", err)
            self._set_metric("status", "idle")
            self.start_btn.setEnabled(True)

    def _stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.state() == QProcess.ProcessState.NotRunning:
            return
        self.proc.terminate()
        self._set_metric("status", "stopping")

    def _read_process_output(self) -> None:
        if self.proc is None:
            return
        data = bytes(self.proc.readAllStandardOutput()).decode("utf-8", errors="replace")
        if not data:
            return
        self._stdout_buf += data
        while "\n" in self._stdout_buf:
            line, self._stdout_buf = self._stdout_buf.split("\n", 1)
            self._append_log(line.rstrip("\r"))
            self._parse_metrics(line)

    def _on_process_finished(self, code: int, _status: QProcess.ExitStatus) -> None:
        if self._stdout_buf:
            tail = self._stdout_buf.rstrip("\r")
            if tail:
                self._append_log(tail)
                self._parse_metrics(tail)
            self._stdout_buf = ""
        self._append_log(f"[process exited with code {code}]")
        self._set_metric("status", "idle")
        self.start_btn.setEnabled(True)

    def _on_process_error(self, _error: QProcess.ProcessError) -> None:
        if self.proc is None:
            return
        self._append_log(f"[process error: {self.proc.errorString()}]")


def build_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(description="Open Training Launcher")


def main() -> None:
    _ = build_parser().parse_args()
    app = QApplication(sys.argv)
    win = TrainLauncherWindow()
    win.show()
    raise SystemExit(app.exec())


if __name__ == "__main__":
    main()
