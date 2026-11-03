from pathlib import Path

p = Path("d:/projects/Test_ScoRe/config.yaml")
raw = p.read_bytes()

OLD = (
    b"score:\r\n"
    b"  # Dynamic-k settings (Layer 1: planner-declared k per window).\r\n"
    b"  # The planner writes EXECUTOR_STEPS: N in its instruction block; the orchestrator\r\n"
    b"  # clamps N to [k_min, k_max] and falls back to k_default when the line is absent.\r\n"
    b"  # Pass --k on the command line to set k_default at runtime.\r\n"
    b"  k_default: 3                           # fallback executor steps when planner doesn't declare k\r\n"
    b"  k_min: 1                               # minimum steps the planner may grant\r\n"
    b"  k_max: 8                               # maximum steps the planner may grant\r\n"
    b"  # Legacy key 'k' is still accepted as an alias for k_default (backward compat)."
)

NEW = (
    b"score:\r\n"
    b"  k: 3                                   # default executor steps per window (override with --k)\r\n"
    b"  k_default: 3                           # fallback executor steps when planner doesn't declare k\r\n"
    b"  k_min: 1                               # minimum steps the planner may grant\r\n"
    b"  k_max: 8                               # maximum steps the planner may grant"
)

assert OLD in raw, "Target block in config.yaml not found!"
raw = raw.replace(OLD, NEW)
p.write_bytes(raw)
print("Updated config.yaml with k key preserved.")
