from pathlib import Path

p = Path("d:/projects/Test_ScoRe/score/prompts.py")
raw = p.read_bytes()

OLD = (
    b"     Choose how many steps the executor gets by including the line EXECUTOR_STEPS: N.\r\n"
    b"     Set N to match the subgoal complexity: fewer for simple lookups (1-2),\r\n"
    b"     more for multi-stage sequences (4-{k_max}). Clamped to [{k_min}, {k_max}].\r\n"
    b"     When omitted, the executor gets {k_default} steps."
)
NEW = (
    b"     Choose how many steps the executor gets by including the line EXECUTOR_STEPS: N.\r\n"
    b"     Set N to match the subgoal complexity: fewer for simple lookups (1-2),\r\n"
    b"     more for multi-stage sequences (4-{k_max}). Clamped to [{k_min}, {k_max}].\r\n"
    b"     When omitted, the executor gets {k} steps."
)

raw = raw.replace(b"{k_max}", b"8").replace(b"{k_min}", b"1").replace(b"{k_default}", b"{k}")
p.write_bytes(raw)
print("Updated prompts.py placeholders cleanly.")
