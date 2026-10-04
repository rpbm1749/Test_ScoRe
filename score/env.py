"""Thin wrapper around an AppWorld task so the agents don't depend on AppWorld directly.

The tests swap this for a fake environment with the same methods.

Windows note: AppWorld stops long-running code with signal.SIGALRM, which only exists on
Linux/macOS. On Windows we switch that off (timeout_seconds=None) and use our own timer
instead: a background thread interrupts the main thread if one execution runs too long
(e.g. a model wrote an infinite loop), and the step is reported as timed out.
"""

from __future__ import annotations

import _thread
import json
import os
import threading
from typing import Any

EXECUTION_TIMEOUT_S = 100  # same as AppWorld's own default
IS_WINDOWS = os.name == "nt"


def appworld_kwargs() -> dict[str, Any]:
    """Extra AppWorld(...) arguments needed on this OS."""
    return {"timeout_seconds": None} if IS_WINDOWS else {}


class _Watchdog:
    """Interrupts the main thread after `seconds` unless cancelled first."""

    def __init__(self, seconds: float) -> None:
        self.fired = False
        self._lock = threading.Lock()
        self._done = False
        self._timer = threading.Timer(seconds, self._fire)
        self._timer.daemon = True

    def _fire(self) -> None:
        with self._lock:
            if self._done:
                return
            self.fired = True
        _thread.interrupt_main()

    def __enter__(self) -> "_Watchdog":
        self._timer.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        with self._lock:
            self._done = True
        self._timer.cancel()


def run_with_timeout(fn, *args: Any, seconds: float = EXECUTION_TIMEOUT_S) -> tuple[bool, Any]:
    """Run fn(*args); returns (timed_out, result). Only needed where SIGALRM is missing."""
    try:
        with _Watchdog(seconds) as dog:
            result = fn(*args)
    except KeyboardInterrupt:
        if dog.fired:
            return True, None
        raise
    if dog.fired:
        # The interrupt may have been caught inside the executed cell (IPython reports it as
        # an error); either way the step ran past the limit.
        return True, result
    return False, result


TIMEOUT_MESSAGE = f"Execution failed. Traceback:\nExecution timed out after {EXECUTION_TIMEOUT_S} seconds."

PROTECTED_NAMES = ("apis", "requester")
RESTORED_NOTE = (
    "\n[Note: this step reassigned `{name}`, which must always refer to the app APIs. It has been "
    "restored. Store results in a different variable name.]"
)


def restore_protected(namespace: dict[str, Any], originals: dict[str, Any]) -> str:
    """Put back objects the model must not overwrite (e.g. `apis = apis.api_docs...`).

    Returns a note to append to the step's output, or "" if nothing was changed."""
    notes = []
    for name, obj in originals.items():
        if obj is not None and namespace.get(name) is not obj:
            namespace[name] = obj
            notes.append(RESTORED_NOTE.format(name=name))
    return "".join(notes)


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
            **appworld_kwargs(),
        )
        task = self.world.task
        self.instruction: str = task.instruction
        self.supervisor: dict[str, Any] = dict(task.supervisor)
        self.app_descriptions: str = json.dumps(
            [{"name": k, "description": v} for k, v in task.app_descriptions.items()], indent=1
        )

    def execute(self, code: str) -> str:
        self.steps += 1
        if not IS_WINDOWS:
            output = self.world.execute(code)
        else:
            timed_out, output = run_with_timeout(self.world.execute, code)
            if timed_out:
                output = TIMEOUT_MESSAGE
        return output + self._guard()

    def _guard(self) -> str:
        shell = getattr(self.world, "shell", None)
        if shell is None:
            return ""
        originals = {"apis": getattr(self.world, "apis", None), "requester": getattr(self.world, "requester", None)}
        return restore_protected(shell.user_ns, originals)

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
