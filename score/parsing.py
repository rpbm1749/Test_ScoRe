"""Pull code, instruction blocks and handback signals out of model replies."""

from __future__ import annotations

import json
import re
from typing import Any

_FULL_CODE = re.compile(r"```(?:python|py)[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_PARTIAL_CODE = re.compile(r"```(?:python|py)[ \t]*\n(.*)$", re.DOTALL | re.IGNORECASE)
_INSTRUCTIONS = re.compile(r"<instructions>(.*?)</instructions>", re.DOTALL | re.IGNORECASE)
_HANDBACK = re.compile(r"^\s*HANDBACK\s*:\s*(.+)$", re.MULTILINE | re.IGNORECASE)


def extract_code(text: str) -> str | None:
    """First ```python block (later blocks are ignored, like AppWorld's ReAct agent)."""
    m = _FULL_CODE.search(text)
    if m:
        code = m.group(1).strip()
        return code or None
    m = _PARTIAL_CODE.search(text)  # reply cut off before the closing fence
    if m and m.group(1).strip():
        return m.group(1).strip()
    return None


def thought_before_code(text: str) -> str:
    idx = text.find("```")
    return (text[:idx] if idx >= 0 else text).strip()


def extract_instructions(text: str) -> str | None:
    m = _INSTRUCTIONS.search(text)
    return m.group(1).strip() if m else None


def strip_instructions(text: str) -> str:
    return _INSTRUCTIONS.sub("", text).strip()


def extract_handback(text: str) -> str | None:
    """Return the handback reason if the reply is a HANDBACK with no code."""
    if extract_code(text):
        return None
    m = _HANDBACK.search(text)
    return m.group(1).strip() if m else None


def calls_complete_task(code: str) -> bool:
    return "complete_task" in code


def is_error_output(output: str) -> bool:
    return output.lstrip().startswith("Execution failed")


def truncate(text: str, max_chars: int) -> str:
    """Keep the head and tail of long outputs (same rule for every mode)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    cut = len(text) - 2 * half
    return f"{text[:half]}\n... [{cut} characters truncated] ...\n{text[-half:]}"


_FENCE = re.compile(r"^```(?:python|py)?\s*\n(.*?)\n?```\s*$", re.DOTALL | re.IGNORECASE)


def code_from_arguments(arguments: Any) -> str:
    """Tool arguments should be JSON {"code": "..."}; also accept raw code or a fenced block."""
    code: Any = arguments
    if isinstance(arguments, str):
        try:
            code = json.loads(arguments)
        except (json.JSONDecodeError, ValueError):
            code = arguments
    if isinstance(code, dict):
        code = code.get("code") or next((v for v in code.values() if isinstance(v, str)), "")
    code = str(code or "").strip()
    m = _FENCE.match(code)
    return m.group(1).strip() if m else code


_ARGS_TAIL = re.compile(r'"arguments"\s*:\s*(.*?)\s*\}?\s*$', re.DOTALL)


def _compiles(code: str) -> bool:
    try:
        compile(code, "<step>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def repair_code(code: str) -> str:
    """Fix the stray quoting that malformed tool calls leave behind, e.g. a trailing `"`.

    Tries a few candidate repairs and returns the first that is valid Python. If none is,
    returns the input unchanged (the step then fails with a syntax error, as it would anyway)."""
    code = code.strip()
    if _compiles(code):
        return code
    candidates = []
    # a JSON string literal, possibly missing its closing quote
    for c in (code, code + '"'):
        if c.startswith('"'):
            try:
                decoded = json.loads(c)
                if isinstance(decoded, str):
                    candidates.append(decoded)
            except (json.JSONDecodeError, ValueError):
                pass
    stripped = code
    for _ in range(3):  # stray quotes / braces at the ends
        stripped = stripped.rstrip().rstrip('"').rstrip("}").rstrip().rstrip('"')
        candidates.append(stripped)
    candidates.append(code.strip('"').strip())
    candidates.append(code.strip('"').strip().encode().decode("unicode_escape", errors="ignore"))
    for c in candidates:
        c = c.strip()
        if c and _compiles(c):
            m = _FENCE.match(c)
            return m.group(1).strip() if m else c
    return code


def code_from_failed_generation(failed: str) -> str | None:
    """Recover the code from a tool call the provider could not parse.

    gpt-oss sometimes calls its built-in `python` tool with raw code instead of JSON, e.g.
    {"name": "python", "arguments": print(apis.api_docs.show_app_descriptions())\n}
    """
    failed = (failed or "").strip()
    if not failed:
        return None
    code: str | None = None
    try:
        obj = json.loads(failed)
        if isinstance(obj, dict) and "arguments" in obj:
            code = code_from_arguments(obj["arguments"]) or None
    except (json.JSONDecodeError, ValueError):
        m = _ARGS_TAIL.search(failed)
        if m:
            code = code_from_arguments(m.group(1)) or None
    return repair_code(code) if code else None
