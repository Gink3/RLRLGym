"""GUI launcher for training with live metric updates."""

from __future__ import annotations

import argparse
import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except Exception as exc:  # pragma: no cover
    raise RuntimeError("Tkinter is required for Training Launcher") from exc

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rlrlgym.themes import (  # noqa: E402
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


class TrainLauncherApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("RLRLGym Training Launcher")

        self._theme_name = load_selected_theme()
        self._theme = get_theme(self._theme_name)

        self.proc: subprocess.Popen[str] | None = None
        self._reader_thread: threading.Thread | None = None
        self._queue: queue.Queue[str] = queue.Queue()

        self._build_ui()
        self._apply_theme()
        self._tick_queue()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root)
        top.pack(fill="x", padx=8, pady=8)

        self.backend_var = tk.StringVar(value="custom")
        self.seed_var = tk.StringVar(value="0")
        self.output_var = tk.StringVar(value="outputs/train/gui_run")
        self.scenario_var = tk.StringVar(value="")
        self.episodes_var = tk.StringVar(value="100")
        self.iterations_var = tk.StringVar(value="50")
        self.max_steps_var = tk.StringVar(value="120")

        ttk.Label(top, text="Backend").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            top,
            textvariable=self.backend_var,
            values=["custom", "rllib"],
            state="readonly",
            width=10,
        ).grid(row=0, column=1, padx=(6, 12), sticky="w")

        ttk.Label(top, text="Seed").grid(row=0, column=2, sticky="w")
        ttk.Entry(top, textvariable=self.seed_var, width=8).grid(row=0, column=3, padx=(6, 12), sticky="w")

        ttk.Label(top, text="Episodes").grid(row=0, column=4, sticky="w")
        ttk.Entry(top, textvariable=self.episodes_var, width=8).grid(row=0, column=5, padx=(6, 12), sticky="w")

        ttk.Label(top, text="Iterations").grid(row=0, column=6, sticky="w")
        ttk.Entry(top, textvariable=self.iterations_var, width=8).grid(row=0, column=7, padx=(6, 12), sticky="w")

        ttk.Label(top, text="Max Steps").grid(row=0, column=8, sticky="w")
        ttk.Entry(top, textvariable=self.max_steps_var, width=8).grid(row=0, column=9, padx=(6, 0), sticky="w")

        row2 = ttk.Frame(self.root)
        row2.pack(fill="x", padx=8, pady=(0, 8))

        ttk.Label(row2, text="Output Dir").pack(side="left")
        ttk.Entry(row2, textvariable=self.output_var, width=46).pack(side="left", padx=(6, 6))
        ttk.Button(row2, text="Browse", command=self._pick_output_dir).pack(side="left")

        row3 = ttk.Frame(self.root)
        row3.pack(fill="x", padx=8, pady=(0, 8))

        ttk.Label(row3, text="Scenario Path").pack(side="left")
        ttk.Entry(row3, textvariable=self.scenario_var, width=46).pack(side="left", padx=(6, 6))
        ttk.Button(row3, text="Browse", command=self._pick_scenario).pack(side="left")

        row4 = ttk.Frame(self.root)
        row4.pack(fill="x", padx=8, pady=(0, 8))

        self.start_btn = ttk.Button(row4, text="Start Training", command=self._start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(row4, text="Stop", command=self._stop)
        self.stop_btn.pack(side="left", padx=(6, 0))

        self.theme_var = tk.StringVar(value=self._theme_name)
        self.settings_button = ttk.Menubutton(row4, text="Settings")
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

        metrics = ttk.LabelFrame(self.root, text="Live Metrics")
        metrics.pack(fill="x", padx=8, pady=(0, 8))

        self.metric_vars = {
            "ret": tk.StringVar(value="-"),
            "win": tk.StringVar(value="-"),
            "surv": tk.StringVar(value="-"),
            "starve": tk.StringVar(value="-"),
            "loss": tk.StringVar(value="-"),
            "eps": tk.StringVar(value="-"),
            "status": tk.StringVar(value="idle"),
        }
        col = 0
        for key, label in [
            ("ret", "Return"),
            ("win", "Win"),
            ("surv", "Survival"),
            ("starve", "Starve"),
            ("loss", "Loss"),
            ("eps", "Epsilon"),
        ]:
            ttk.Label(metrics, text=label).grid(row=0, column=col, sticky="w", padx=6, pady=4)
            ttk.Label(metrics, textvariable=self.metric_vars[key]).grid(
                row=1, column=col, sticky="w", padx=6, pady=(0, 6)
            )
            col += 1
        ttk.Label(metrics, text="Status").grid(row=0, column=col, sticky="w", padx=6, pady=4)
        ttk.Label(metrics, textvariable=self.metric_vars["status"]).grid(
            row=1, column=col, sticky="w", padx=6, pady=(0, 6)
        )

        self.log_text = tk.Text(self.root, height=24, font=("Courier", 10), wrap="none")
        self.log_text.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log_text.configure(state="disabled")

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
        style.configure("TLabelframe", background=t["panel"], foreground=t["text"])
        style.configure("TLabelframe.Label", background=t["panel"], foreground=t["text"])
        style.configure("TButton", background=t["panel_alt"], foreground=t["text"])
        style.map("TButton", background=[("active", t["primary_hover"])])
        style.configure("TMenubutton", background=t["panel_alt"], foreground=t["text"])
        style.configure(
            "TEntry",
            fieldbackground=t["panel_alt"],
            background=t["panel_alt"],
            foreground=t["text"],
            bordercolor=t["border"],
        )
        style.configure(
            "TCombobox",
            fieldbackground=t["panel_alt"],
            background=t["panel_alt"],
            foreground=t["text"],
            bordercolor=t["border"],
        )
        self.log_text.configure(
            bg=t["panel_alt"], fg=t["text"], insertbackground=t["text"]
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

    def _pick_output_dir(self) -> None:
        p = filedialog.askdirectory(title="Select output directory", initialdir="outputs")
        if p:
            self.output_var.set(str(Path(p)))

    def _pick_scenario(self) -> None:
        p = filedialog.askdirectory(title="Select scenario directory", initialdir="data/scenarios")
        if p:
            self.scenario_var.set(str(Path(p)))

    def _append_log(self, line: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert(tk.END, line + "\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state="disabled")

    def _parse_metrics(self, line: str) -> None:
        m = CUSTOM_PROGRESS_RE.search(line)
        if not m:
            return
        for k in ("ret", "win", "surv", "starve", "loss", "eps"):
            self.metric_vars[k].set(m.group(k))

    def _build_command(self) -> list[str]:
        cmd = ["python3", "-m", "train", "--backend", self.backend_var.get().strip() or "custom"]
        cmd += ["--seed", self.seed_var.get().strip() or "0"]
        cmd += ["--output-dir", self.output_var.get().strip() or "outputs/train/gui_run"]
        max_steps = self.max_steps_var.get().strip()
        if max_steps:
            cmd += ["--max-steps", max_steps]
        scenario = self.scenario_var.get().strip()
        if scenario:
            cmd += ["--scenario-path", scenario]
        if self.backend_var.get().strip() == "custom":
            cmd += ["--episodes", self.episodes_var.get().strip() or "100"]
        else:
            cmd += ["--iterations", self.iterations_var.get().strip() or "50"]
        return cmd

    def _start(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            messagebox.showwarning("Training", "Training is already running.")
            return
        cmd = self._build_command()
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path(__file__).resolve().parents[1]),
                env=env,
            )
        except Exception as exc:
            messagebox.showerror("Failed to start", str(exc))
            return

        self.metric_vars["status"].set("running")
        self._append_log("$ " + " ".join(cmd))
        self.start_btn.state(["disabled"])

        def _reader() -> None:
            assert self.proc is not None
            if self.proc.stdout is None:
                return
            for line in self.proc.stdout:
                self._queue.put(line.rstrip("\n"))
            rc = self.proc.wait()
            self._queue.put(f"[process exited with code {rc}]")
            self._queue.put("__PROCESS_DONE__")

        self._reader_thread = threading.Thread(target=_reader, daemon=True)
        self._reader_thread.start()

    def _stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        self.proc.terminate()
        self.metric_vars["status"].set("stopping")

    def _tick_queue(self) -> None:
        try:
            while True:
                line = self._queue.get_nowait()
                if line == "__PROCESS_DONE__":
                    self.metric_vars["status"].set("idle")
                    self.start_btn.state(["!disabled"])
                    continue
                self._append_log(line)
                self._parse_metrics(line)
        except queue.Empty:
            pass
        self.root.after(100, self._tick_queue)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Open Training Launcher")
    return p


def main() -> None:
    _ = build_parser().parse_args()
    root = tk.Tk()
    app = TrainLauncherApp(root)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    main()
