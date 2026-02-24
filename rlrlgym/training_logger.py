"""Training logger and dashboard report generation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List

from .constants import ACTION_NAMES


@dataclass
class EpisodeSummary:
    episode: int
    steps: int
    team_return: float
    per_agent_return: Dict[str, float]
    win: bool
    mean_survival_time: float
    cause_of_death: Dict[str, str]
    action_counts: Dict[str, int]


@dataclass
class TrainingLogger:
    output_dir: str = "outputs"
    episode_summaries: List[EpisodeSummary] = field(default_factory=list)
    _episode: int = 0
    _returns: Dict[str, float] = field(default_factory=dict)
    _death_cause: Dict[str, str] = field(default_factory=dict)
    _survival_steps: Dict[str, int] = field(default_factory=dict)
    _action_counts: Dict[str, int] = field(default_factory=dict)

    @staticmethod
    def _action_keys() -> List[str]:
        return [ACTION_NAMES[k] for k in sorted(ACTION_NAMES.keys())]

    def start_episode(self, agent_ids: List[str]) -> None:
        self._episode += 1
        self._returns = {aid: 0.0 for aid in agent_ids}
        self._death_cause = {aid: "alive" for aid in agent_ids}
        self._survival_steps = {aid: 0 for aid in agent_ids}
        self._action_counts = {name: 0 for name in self._action_keys()}

    def log_step(
        self,
        rewards: Dict[str, float],
        terminations: Dict[str, bool],
        truncations: Dict[str, bool],
        info: Dict[str, Dict[str, object]],
        actions: Dict[str, int] | None = None,
    ) -> None:
        for aid, reward in rewards.items():
            self._returns[aid] = self._returns.get(aid, 0.0) + float(reward)

        for aid in self._survival_steps:
            done = bool(terminations.get(aid, False) or truncations.get(aid, False))
            if not done:
                self._survival_steps[aid] += 1
            elif self._death_cause[aid] == "alive":
                events = info.get(aid, {}).get("events", [])
                if "death" in events and "starve_tick" in events:
                    self._death_cause[aid] = "starvation"
                elif "death" in events:
                    self._death_cause[aid] = "damage_or_other"
                elif truncations.get(aid, False):
                    self._death_cause[aid] = "timeout_alive"
                else:
                    self._death_cause[aid] = "unknown"

        if actions:
            for action in actions.values():
                action_name = ACTION_NAMES.get(int(action), f"unknown_{int(action)}")
                self._action_counts[action_name] = self._action_counts.get(action_name, 0) + 1

    def end_episode(self, step_count: int, alive_agents: Dict[str, bool]) -> EpisodeSummary:
        team_return = sum(self._returns.values())
        win = any(alive_agents.values())
        mean_survival = sum(self._survival_steps.values()) / max(1, len(self._survival_steps))
        summary = EpisodeSummary(
            episode=self._episode,
            steps=step_count,
            team_return=team_return,
            per_agent_return=dict(self._returns),
            win=win,
            mean_survival_time=mean_survival,
            cause_of_death=dict(self._death_cause),
            action_counts=dict(self._action_counts),
        )
        self.episode_summaries.append(summary)
        return summary

    def aggregate_metrics(self) -> Dict[str, object]:
        if not self.episode_summaries:
            return {
                "episodes": 0,
                "win_rate": 0.0,
                "mean_team_return": 0.0,
                "mean_survival_time": 0.0,
                "cause_of_death_histogram": {},
                "action_histogram": {},
            }

        wins = sum(1 for e in self.episode_summaries if e.win)
        mean_return = sum(e.team_return for e in self.episode_summaries) / len(self.episode_summaries)
        mean_survival = sum(e.mean_survival_time for e in self.episode_summaries) / len(self.episode_summaries)

        cod: Dict[str, int] = {}
        for e in self.episode_summaries:
            for cause in e.cause_of_death.values():
                cod[cause] = cod.get(cause, 0) + 1
        actions: Dict[str, int] = {}
        for e in self.episode_summaries:
            for action_name, count in e.action_counts.items():
                actions[action_name] = actions.get(action_name, 0) + int(count)

        return {
            "episodes": len(self.episode_summaries),
            "win_rate": wins / len(self.episode_summaries),
            "mean_team_return": mean_return,
            "mean_survival_time": mean_survival,
            "cause_of_death_histogram": cod,
            "action_histogram": actions,
        }

    def write_outputs(self) -> Dict[str, str]:
        out = Path(self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        jsonl_path = out / "episodes.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for e in self.episode_summaries:
                f.write(json.dumps(asdict(e)) + "\n")

        csv_path = out / "episodes.csv"
        all_agents = sorted({aid for e in self.episode_summaries for aid in e.per_agent_return})
        all_actions = sorted({k for e in self.episode_summaries for k in e.action_counts.keys()})
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["episode", "steps", "team_return", "win", "mean_survival_time"] + [
                f"return_{aid}" for aid in all_agents
            ] + [f"action_{action_name}" for action_name in all_actions]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in self.episode_summaries:
                row = {
                    "episode": e.episode,
                    "steps": e.steps,
                    "team_return": round(e.team_return, 5),
                    "win": int(e.win),
                    "mean_survival_time": round(e.mean_survival_time, 5),
                }
                for aid in all_agents:
                    row[f"return_{aid}"] = round(e.per_agent_return.get(aid, 0.0), 5)
                for action_name in all_actions:
                    row[f"action_{action_name}"] = int(e.action_counts.get(action_name, 0))
                writer.writerow(row)

        html_path = out / "dashboard.html"
        html_path.write_text(self._build_dashboard_html(), encoding="utf-8")

        summary_path = out / "summary.json"
        summary_path.write_text(json.dumps(self.aggregate_metrics(), indent=2), encoding="utf-8")

        return {
            "jsonl": str(jsonl_path),
            "csv": str(csv_path),
            "html": str(html_path),
            "summary": str(summary_path),
        }

    def _build_dashboard_html(self) -> str:
        data = [asdict(e) for e in self.episode_summaries]
        aggregate = self.aggregate_metrics()
        returns = [e["team_return"] for e in data]

        return f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>RLRLGym Training Dashboard</title>
  <style>
    :root {{
      --bg: #1f232b;
      --bg-alt: #262b35;
      --surface: #2d3340;
      --surface-soft: #343b49;
      --text: #f1f4fb;
      --muted: #b7bfd3;
      --border: #4b5468;
      --royal-purple: #6a3fe8;
      --royal-purple-2: #7d5cf0;
      --teal-accent: #39d0c3;
      --gold-accent: #f2c14e;
      --rose-accent: #ef7fa8;
    }}
    body {{
      font-family: -apple-system, Segoe UI, sans-serif;
      margin: 24px;
      background: radial-gradient(1200px 600px at 10% -10%, #312554 0%, var(--bg) 45%);
      color: var(--text);
    }}
    h1 {{ letter-spacing: 0.3px; }}
    h3 {{ color: var(--teal-accent); margin-top: 0; }}
    .card {{
      background: linear-gradient(180deg, var(--surface) 0%, var(--bg-alt) 100%);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px;
      margin-bottom: 16px;
      box-shadow: 0 10px 26px rgba(12, 10, 20, 0.35);
    }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{
      border: 1px solid var(--border);
      padding: 6px 8px;
      text-align: left;
      font-size: 13px;
    }}
    th {{
      background: rgba(106, 63, 232, 0.24);
      color: #efe9ff;
    }}
    tr:nth-child(even) td {{ background: rgba(255, 255, 255, 0.02); }}
    .metric {{
      display: inline-block;
      margin-right: 16px;
      margin-bottom: 4px;
      font-weight: 700;
      color: var(--gold-accent);
    }}
    pre {{
      margin: 0;
      padding: 10px;
      border-radius: 8px;
      background: rgba(106, 63, 232, 0.12);
      border: 1px solid rgba(125, 92, 240, 0.35);
      color: #e8deff;
      overflow-x: auto;
    }}
    .bar {{
      height: 18px;
      border-radius: 6px;
      background: linear-gradient(90deg, var(--royal-purple), var(--royal-purple-2), var(--rose-accent));
      margin-bottom: 8px;
      box-shadow: 0 0 0 1px rgba(239, 127, 168, 0.2);
    }}
    .hist-label {{ color: var(--muted); margin-bottom: 2px; }}
  </style>
</head>
<body>
  <h1>RLRLGym Training Dashboard</h1>
  <div class=\"card\">
    <div class=\"metric\">Episodes: {aggregate['episodes']}</div>
    <div class=\"metric\">Win rate: {aggregate['win_rate']:.3f}</div>
    <div class=\"metric\">Mean team return: {aggregate['mean_team_return']:.3f}</div>
    <div class=\"metric\">Mean survival time: {aggregate['mean_survival_time']:.3f}</div>
  </div>
  <div class=\"card\">
    <h3>Episode Return Curve</h3>
    <pre>{returns}</pre>
  </div>
  <div class=\"card\">
    <h3>Cause of Death Histogram</h3>
    {''.join([f'<div class=\"hist-label\">{k}: {v}</div><div class=\"bar\" style=\"width:{20 + v * 20}px\"></div>' for k, v in aggregate['cause_of_death_histogram'].items()])}
  </div>
  <div class=\"card\">
    <h3>Total Action Usage</h3>
    <pre>{json.dumps(aggregate["action_histogram"], indent=2)}</pre>
  </div>
  <div class=\"card\">
    <h3>Episode Table</h3>
    <table>
      <thead><tr><th>Episode</th><th>Steps</th><th>Team Return</th><th>Win</th><th>Mean Survival</th><th>Cause Of Death</th><th>Action Counts</th></tr></thead>
      <tbody>
        {''.join([f'<tr><td>{e["episode"]}</td><td>{e["steps"]}</td><td>{e["team_return"]:.3f}</td><td>{int(e["win"])}</td><td>{e["mean_survival_time"]:.3f}</td><td>{e["cause_of_death"]}</td><td>{e["action_counts"]}</td></tr>' for e in data])}
      </tbody>
    </table>
  </div>
</body>
</html>"""
