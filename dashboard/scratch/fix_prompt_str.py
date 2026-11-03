from pathlib import Path

p = Path("d:/projects/Test_ScoRe/score/prompts.py")
raw = p.read_bytes()

OLD = (
    b"for the executor's next window.\r\n"
    b"     Choose how many steps the executor gets by including the line EXECUTOR_STEPS: N.\r\n"
    b"     Set N to match the subgoal complexity: fewer for simple lookups (1-2),\r\n"
    b"     more for multi-stage sequences (4-{8}). Clamped to [{1}, {8}].\r\n"
    b"     When omitted, the executor gets {{k}} steps."
)
NEW = (
    b"for the executor's next window.\r\n"
    b"     Choose how many steps the executor gets by including the line EXECUTOR_STEPS: N.\r\n"
    b"     Set N to match the subgoal complexity: fewer for simple lookups (1-2),\r\n"
    b"     more for multi-stage sequences (4-8). Clamped to [1, 8].\r\n"
    b"     When omitted, the executor gets {k} steps."
)

assert OLD in raw, "Target string not found!"
raw = raw.replace(OLD, NEW)
p.write_bytes(raw)
print("Successfully fixed prompt string literals!")
