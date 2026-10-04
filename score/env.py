"""Thin wrapper around an AppWorld task so the agents don't depend on AppWorld directly.

The tests swap this for a fake environment with the same methods.
"""

from __future__ import annotations

import json
from typing import Any


class AppWorldEnv:
    def __init__(self, task_id: str, experiment_name: str, max_steps: int) -> None:
        from appworld import AppWorld  # imported lazily so tests run without AppWorld data

        self.task_id = task_id
        self.max_steps = max_steps
        self.steps = 0
        # max_interactions is a hard backstop inside AppWorld; our own max_steps is the real budget.
        self.world = AppWorld(
            task_id=task_id,
            experiment_name=experiment_name,
            max_interactions=max_steps + 5,
        )
        task = self.world.task
        self.instruction: str = task.instruction
        self.supervisor: dict[str, Any] = dict(task.supervisor)
        self.app_descriptions: str = json.dumps(
            [{"name": k, "description": v} for k, v in task.app_descriptions.items()], indent=1
        )

    def execute(self, code: str) -> str:
        self.steps += 1
        return self.world.execute(code)

    def completed(self) -> bool:
        return bool(self.world.task_completed())

    @property
    def out_of_steps(self) -> bool:
        return self.steps >= self.max_steps

    def evaluate(self) -> dict[str, Any]:
        tracker = self.world.evaluate()
        return {
            "success": bool(tracker.success),
            "pass_percentage": float(tracker.pass_percentage),
            "pass_count": int(tracker.pass_count),
            "num_tests": int(tracker.num_tests),
        }

    def close(self) -> None:
        try:
            self.world.close()
        except Exception:
            pass
