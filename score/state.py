"""State objects for ScoRe and the trajectory log shared by every mode.

GlobalState  - what the planner (frontier model) sees: its own running conversation,
               which holds the task, every instruction block it wrote, and every step
               (its own and the executor's) with outputs truncated by the same rule the
               single-model baseline uses.
LocalState   - what the executor (cheap model) sees in one window: the task, the planner's
               latest instruction block, the planner's latest step + output, and only its
               own steps in this window. It is rebuilt from scratch for every window.
Trajectory   - the full log of every LLM call and every environment step, saved to JSON.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

from .llm import Usage


@dataclass
class StepRecord:
    step: int                 # environment step number (1-based)
    actor: str                # "agent" (single-model), "planner" or "executor"
    window: int | None        # ScoRe window index (None for single-model runs)
    thought: str
    code: str
    output: str               # raw output from AppWorld
    output_shown: str         # what the models saw (after truncation)
    error: bool


@dataclass
class CallRecord:
    role: str
    window: int | None
    produced_code: bool
    note: str
    usage: dict[str, Any]


@dataclass
class WindowRecord:
    window: int
    instructions: str | None
    planner_step: int | None
    executor_steps: list[int] = field(default_factory=list)
    handback_reason: str = ""


@dataclass
class GlobalState:
    task_line: str
    planner_messages: list[dict[str, str]] = field(default_factory=list)
    windows: list[WindowRecord] = field(default_factory=list)


@dataclass
class LocalState:
    window: int
    instructions: str
    planner_code: str
    planner_output: str
    messages: list[dict[str, str]] = field(default_factory=list)
    steps_taken: int = 0
    consecutive_errors: int = 0
    step_records: list[StepRecord] = field(default_factory=list)


class Trajectory:
    def __init__(self, task_id: str, mode: str, k: int | None) -> None:
        self.task_id = task_id
        self.mode = mode
        self.k = k
        self.steps: list[StepRecord] = []
        self.calls: list[CallRecord] = []
        self.usage: dict[str, Usage] = {}
        self.t0 = time.time()

    def add_call(self, role: str, usage: Usage, window: int | None, produced_code: bool, note: str = "") -> None:
        self.calls.append(CallRecord(role, window, produced_code, note, usage.to_dict()))
        self.usage.setdefault(role, Usage()).add(usage)

    def add_step(self, rec: StepRecord) -> None:
        self.steps.append(rec)

    def steps_by_actor(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.steps:
            out[s.actor] = out.get(s.actor, 0) + 1
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "mode": self.mode,
            "k": self.k,
            "wall_time_s": round(time.time() - self.t0, 1),
            "usage": {r: u.to_dict() for r, u in self.usage.items()},
            "steps_by_actor": self.steps_by_actor(),
            "steps": [asdict(s) for s in self.steps],
            "calls": [asdict(c) for c in self.calls],
        }
