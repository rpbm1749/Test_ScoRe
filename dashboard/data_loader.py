"""Data loader and helper functions for ScoRe Streamlit Dashboard.

Scans the runs/ directory, parses JSON/JSONL/YAML logs, and prepares dataframes
and structured objects for visualization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import yaml


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
DATA_TASKS_DIR = ROOT / "data" / "tasks"


def get_available_runs() -> list[str]:
    """Return a sorted list of run directory names found in runs/."""
    if not RUNS_DIR.exists():
        return []
    runs = [d.name for d in RUNS_DIR.iterdir() if d.is_dir() and (d / "results.jsonl").exists()]
    return sorted(runs, reverse=True)


def load_run_results(run_name: str) -> list[dict[str, Any]]:
    """Load results.jsonl for a given run."""
    results_file = RUNS_DIR / run_name / "results.jsonl"
    if not results_file.exists():
        return []
    records = []
    with open(results_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def load_run_config(run_name: str) -> dict[str, Any]:
    """Load config_used.yaml for a given run."""
    config_file = RUNS_DIR / run_name / "config_used.yaml"
    if not config_file.exists():
        return {}
    try:
        with open(config_file, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def load_task_trajectory(run_name: str, task_id: str) -> dict[str, Any] | None:
    """Load <task_id>.json trajectory for a given run."""
    traj_file = RUNS_DIR / run_name / f"{task_id}.json"
    if not traj_file.exists():
        return None
    try:
        with open(traj_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_task_specs(task_id: str) -> dict[str, Any]:
    """Load specs.json from data/tasks/<task_id>/specs.json if available."""
    spec_file = DATA_TASKS_DIR / task_id / "specs.json"
    if not spec_file.exists():
        return {}
    try:
        with open(spec_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
