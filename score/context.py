"""Context the orchestrator builds for the models, independent of any particular task suite.

ScoRe wipes the executor's context between windows, but two things persist that it cannot see:

  1. the environment's *state* (for AppWorld: the variables in the Python session), and
  2. *reference material* already looked up (for AppWorld: API documentation).

Without them the executor re-discovers things every window and guesses names. The core only
defines two optional hooks; each environment adapter decides what they mean:

  env.state_summary() -> str
      A short description of the current state. Shown to the executor (and, in the notes
      view, to the planner) at the start of every turn.

  env.reference_from_step(code, output) -> list[(key, text)]
      Called after every executed step. Returns compact reference entries worth keeping
      (e.g. an API signature). A later entry with the same key replaces the earlier one.

  env.reference_for_text(text) -> list[(key, text)]
      Reference entries for things named in free text, e.g. APIs the planner's instructions
      mention, so the executor has them before its first step.

  env.validate(code) -> str | None
      Checked before every step in every mode. Returns a description of what would go wrong
      (e.g. an API parameter that doesn't exist, an undefined variable), or None. A rejected
      step isn't run and doesn't use an environment step; the model gets the message instead.

An environment without these hooks still works; those sections are simply left empty.
Everything here is built by code, not written by a model, so it can't be misremembered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SharedContext:
    reference: dict[str, str] = field(default_factory=dict)
    max_reference_chars: int = 8000

    def record_step(self, env: Any, code: str, output: str) -> None:
        hook = getattr(env, "reference_from_step", None)
        if hook is None:
            return
        try:
            entries = hook(code, output) or []
        except Exception:  # a reference parser must never break a run
            return
        for key, text in entries:
            self.reference.pop(key, None)  # re-insert so the newest entries come last
            self.reference[key] = text

    def record_text(self, env: Any, text: str) -> None:
        hook = getattr(env, "reference_for_text", None)
        if hook is None or not text:
            return
        try:
            entries = hook(text) or []
        except Exception:
            return
        for key, entry in entries:
            self.reference.pop(key, None)
            self.reference[key] = entry

    def reference_text(self) -> str:
        """All entries, oldest dropped first if over the size cap."""
        lines = list(self.reference.values())
        while lines and sum(len(x) + 1 for x in lines) > self.max_reference_chars:
            lines.pop(0)
        return "\n".join(lines) if lines else "(nothing looked up yet)"

    @staticmethod
    def state_text(env: Any) -> str:
        hook = getattr(env, "state_summary", None)
        if hook is None:
            return "(not available for this environment)"
        try:
            return hook() or "(empty)"
        except Exception:
            return "(could not be read)"


def validate_step(env: Any, code: str) -> str | None:
    """The environment's pre-execution check, if it has one. Never raises."""
    hook = getattr(env, "validate", None)
    if hook is None:
        return None
    try:
        return hook(code)
    except Exception:
        return None
