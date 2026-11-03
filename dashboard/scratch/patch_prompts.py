"""Patch score/prompts.py to support dynamic-k (Layer 1).
Replaces the fixed-k planner prompt block with the dynamic-k version.
Run: python dashboard/scratch/patch_prompts.py
"""
from pathlib import Path

path = Path("d:/projects/Test_ScoRe/score/prompts.py")
raw = path.read_bytes()

OLD = (
    b"  2. `executor_instructions`: the instruction block for the executor's next window of up to {k} steps.\r\n"
    b"\r\n"
    b"The executor does NOT see this conversation. It only sees the task, your latest instruction block,\r\n"
    b"your latest step and its output, and its own steps in the current window. So your instruction block\r\n"
    b"must contain everything it needs: exact variable names that already exist in the REPL (e.g. access\r\n"
    b"tokens), relevant IDs/values you have found, and API names to use.\r\n"
    b"\r\n"
    b"Write executor_instructions in exactly this format:\r\n"
    b"\r\n"
    b"SUBGOAL: <what the executor should achieve in its window>\r\n"
    b"STEPS:\r\n"
    b"1. <concrete step>\r\n"
    b"2. ...\r\n"
    b"KNOWN: <REPL variables and facts the executor needs, e.g. `spotify_access_token` holds a valid token>\r\n"
    b"SUCCESS CHECK: <how the executor can tell the subgoal is done>\r\n"
    b"HAND BACK IF: <conditions under which it should stop early and return control to you>"
)

NEW = (
    b"  2. `executor_instructions`: the instruction block for the executor's next window.\r\n"
    b"     Choose how many steps the executor gets by including the line EXECUTOR_STEPS: N.\r\n"
    b"     Set N to match the subgoal complexity: fewer for simple lookups (1-2),\r\n"
    b"     more for multi-stage sequences (4-{{k_max}}). Clamped to [{{k_min}}, {{k_max}}].\r\n"
    b"     When omitted, the executor gets {{k_default}} steps.\r\n"
    b"\r\n"
    b"The executor does NOT see this conversation. It only sees the task, your latest instruction block,\r\n"
    b"your latest step and its output, and its own steps in the current window. So your instruction block\r\n"
    b"must contain everything it needs: exact variable names that already exist in the REPL (e.g. access\r\n"
    b"tokens), relevant IDs/values you have found, and API names to use.\r\n"
    b"\r\n"
    b"Write executor_instructions in exactly this format:\r\n"
    b"\r\n"
    b"SUBGOAL: <what the executor should achieve in its window>\r\n"
    b"EXECUTOR_STEPS: <how many steps you are granting, e.g. 3>\r\n"
    b"STEPS:\r\n"
    b"1. <concrete step>\r\n"
    b"2. ...\r\n"
    b"KNOWN: <REPL variables and facts the executor needs, e.g. `spotify_access_token` holds a valid token>\r\n"
    b"SUCCESS CHECK: <how the executor can tell the subgoal is done>\r\n"
    b"HAND BACK IF: <conditions under which it should stop early and return control to you>"
)

assert OLD in raw, f"OLD block not found in {path}. File may have changed."
count = raw.count(OLD)
assert count == 1, f"Expected 1 occurrence, found {count}"

patched = raw.replace(OLD, NEW, 1)
path.write_bytes(patched)
print(f"prompts.py patched successfully ({len(OLD)} -> {len(NEW)} bytes).")
