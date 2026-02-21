"""Training logger and dashboard report generation."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List


@dataclass
class EpisodeSummary:
    episode: int
    steps: int
    team_return: float
    per_agent_return: Dict[str, float]
    win: bool
    mean_survival_time: float
    cause_of_death: Dict[str, str]


@dataclass
class TrainingLogger:
    output_dir: str = "outputs"
    episode_summaries: List[EpisodeSummary] = field(default_factory=list)
    _episode: int = 0
    _returns: Dict[str, float] = field(default_factory=dict)
    _death_cause: Dict[str, str] = field(default_factory=dict)
    _survival_steps: Dict[str, int] = field(default_factory=dict)

    def start_episode(self, agent_ids: List[str]) -> None:
        self._episode += 1
        self._returns = {aid: 0.0 for aid in agent_ids}
        self._death_cause = {aid: "alive" for aid in agent_ids}
        self._survival_steps = {aid: 0 for aid in agent_ids}

    def log_step(
        self,
        rewards: Dict[str, float],
        terminations: Dict[str, bool],
        truncations: Dict[str, bool],
        info: Dict[str, Dict[str, object]],
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
            }

        wins = sum(1 for e in self.episode_summaries if e.win)
        mean_return = sum(e.team_return for e in self.episode_summaries) / len(self.episode_summaries)
        mean_survival = sum(e.mean_survival_time for e in self.episode_summaries) / len(self.episode_summaries)

        cod: Dict[str, int] = {}
        for e in self.episode_summaries:
            for cause in e.cause_of_death.values():
                cod[cause] = cod.get(cause, 0) + 1

        return {
            "episodes": len(self.episode_summaries),
            "win_rate": wins / len(self.episode_summaries),
            "mean_team_return": mean_return,
            "mean_survival_time": mean_survival,
            "cause_of_death_histogram": cod,
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
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["episode", "steps", "team_return", "win", "mean_survival_time"] + [
                f"return_{aid}" for aid in all_agents
            ]
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
    body {{ font-family: -apple-system, Segoe UI, sans-serif; margin: 24px; background: #f7f9fb; color: #1f2937; }}
    .card {{ background: white; border: 1px solid #d1d5db; border-radius: 8px; padding: 14px; margin-bottom: 16px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; font-size: 13px; }}
    .metric {{ display: inline-block; margin-right: 16px; font-weight: 600; }}
    .bar {{ height: 18px; background: #60a5fa; margin-bottom: 4px; }}
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
    {''.join([f'<div>{k}: {v}<div class="bar" style="width:{20 + v * 20}px"></div></div>' for k, v in aggregate['cause_of_death_histogram'].items()])}
  </div>
  <div class=\"card\">
    <h3>Episode Table</h3>
    <table>
      <thead><tr><th>Episode</th><th>Steps</th><th>Team Return</th><th>Win</th><th>Mean Survival</th><th>Cause Of Death</th></tr></thead>
      <tbody>
        {''.join([f'<tr><td>{e["episode"]}</td><td>{e["steps"]}</td><td>{e["team_return"]:.3f}</td><td>{int(e["win"])}</td><td>{e["mean_survival_time"]:.3f}</td><td>{e["cause_of_death"]}</td></tr>' for e in data])}
      </tbody>
    </table>
  </div>
</body>
</html>"""
