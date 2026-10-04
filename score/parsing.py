"""Pull code, instruction blocks and handback signals out of model replies."""

from __future__ import annotations

import re

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


def is_error_output(output: str) -> bool:
    return output.lstrip().startswith("Execution failed")


def truncate(text: str, max_chars: int) -> str:
    """Keep the head and tail of long outputs (same rule for every mode)."""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    half = max_chars // 2
    cut = len(text) - 2 * half
    return f"{text[:half]}\n... [{cut} characters truncated] ...\n{text[-half:]}"
