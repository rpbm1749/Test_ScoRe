"""Turning a model reply into an action, and feeding the result back.

gpt-oss models are trained to run code through a tool, so we give every model one tool,
`execute_python`, in every mode. A reply's code comes from:
  1. its first execute_python tool call (the normal case), or
  2. a ```python block in its text (fallback, in case a model writes code instead of calling).
The output then goes back as a `tool` message (case 1) or a user message (case 2), which is
the format the API requires.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from . import prompts as P
from .llm import ChatResult
from .parsing import code_from_arguments, extract_code, extract_instructions, strip_instructions, thought_before_code

TOOL_NAME = "execute_python"

TOOLS: list[dict[str, Any]] = [{
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Run Python code in the task's stateful REPL, where the `apis` object is available. "
            "Variables persist between calls. Returns whatever the code prints, or the error traceback."
        ),
        "parameters": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": "The Python code to run for this step."}},
            "required": ["code"],
        },
    },
}]

# The planner's version of the tool also carries its instructions for the executor. Reasoning
# models often reply with nothing but a tool call (their thinking goes to a separate channel),
# so free text next to the call is unreliable; a required tool argument is not.
PLANNER_TOOLS: list[dict[str, Any]] = [{
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": (
            "Run Python code in the task's stateful REPL (where `apis` is available) as your step, and hand the "
            "executor its instructions for the next window. Returns what the code prints, or the error traceback."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "The Python code for your one step."},
                "executor_instructions": {
                    "type": "string",
                    "description": (
                        "Instructions for the executor's next window, with the lines SUBGOAL, STEPS, KNOWN, "
                        "SUCCESS CHECK and HAND BACK IF. Use an empty string only on your final step "
                        "(the one that calls apis.supervisor.complete_task)."
                    ),
                },
            },
            "required": ["code", "executor_instructions"],
        },
    },
}]


@dataclass
class Action:
    code: str | None
    thought: str
    via_tool: bool
    tool_call_id: str | None
    assistant_message: dict[str, Any]
    extra_tool_messages: list[dict[str, Any]] = field(default_factory=list)
    arguments: dict[str, Any] = field(default_factory=dict)   # parsed tool-call arguments, if JSON


def parse_action(res: ChatResult, text_has_instructions: bool = False) -> Action:
    content = res.content or ""
    text = strip_instructions(content) if text_has_instructions else content
    assistant: dict[str, Any] = {"role": "assistant", "content": content, **res.assistant_extra}

    calls = [c for c in res.tool_calls if c.get("type", "function") == "function"]
    if calls:
        assistant["tool_calls"] = res.tool_calls
        first = calls[0]
        raw_args = first.get("function", {}).get("arguments", "")
        code = code_from_arguments(raw_args)
        try:
            parsed = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except (json.JSONDecodeError, ValueError):
            parsed = {}
        extra = [
            {"role": "tool", "tool_call_id": c.get("id", ""), "content": P.EXTRA_CALL_NOT_RUN}
            for c in res.tool_calls
            if c is not first
        ]
        return Action(code or None, text.strip(), True, first.get("id", ""), assistant, extra,
                      parsed if isinstance(parsed, dict) else {})

    code = extract_code(text)
    return Action(code, thought_before_code(text), False, None, assistant)


def append_reply(messages: list[dict[str, Any]], action: Action) -> None:
    messages.append(action.assistant_message)


def observation_messages(action: Action, shown_output: str) -> list[dict[str, Any]]:
    """Messages that carry a step's output back to the model that took it."""
    if action.via_tool:
        return [{"role": "tool", "tool_call_id": action.tool_call_id, "content": shown_output}, *action.extra_tool_messages]
    return [{"role": "user", "content": P.OBSERVATION.format(output=shown_output)}]


def nudge_messages(action: Action, nudge: str) -> list[dict[str, Any]]:
    """Messages asking the model to try again (answers any tool calls first, as the API requires)."""
    if action.via_tool:
        return [{"role": "tool", "tool_call_id": action.tool_call_id, "content": P.EMPTY_CODE}, *action.extra_tool_messages,
                {"role": "user", "content": nudge}]
    return [{"role": "user", "content": nudge}]


def planner_instructions(res: ChatResult, act: Action) -> str | None:
    """The planner's executor instructions: from the tool argument, else an <instructions> block
    in the reply text, else one in its reasoning."""
    arg = act.arguments.get("executor_instructions")
    if isinstance(arg, str) and arg.strip():
        return extract_instructions(arg) or arg.strip()
    return extract_instructions(res.content or "") or extract_instructions(res.reasoning or "")
