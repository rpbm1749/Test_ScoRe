from pathlib import Path

path = Path("summarize.py")
raw = path.read_bytes()

# 1. Target around trimmed_calls
target1 = (
    b'            "trimmed_calls": sum(r.get("frontier_trimmed_calls", 0) + r.get("cheap_trimmed_calls", 0) for r in rows),\r\n'
    b'        }'
)

replacement1 = (
    b'            "trimmed_calls": sum(r.get("frontier_trimmed_calls", 0) + r.get("cheap_trimmed_calls", 0) for r in rows),\r\n'
    b'            "avg_granted_k": round(sum(r.get("avg_granted_k", 0) for r in rows) / n, 2),\r\n'
    b'            "avg_used_steps": round(sum(r.get("avg_used_steps", 0) for r in rows) / n, 2),\r\n'
    b'            "pct_declared": round(sum(r.get("pct_declared", 0) for r in rows) / n, 3),\r\n'
    b'            "pct_stale": round(sum(r.get("pct_stale", 0) for r in rows) / n, 3),\r\n'
    b'        }'
)

assert target1 in raw, "target1 not found in summarize.py"
raw = raw.replace(target1, replacement1, 1)

# 2. Target before cols = list(out_rows[0].keys())
target2 = b'    cols = list(out_rows[0].keys())'
replacement2 = (
    b'    if not out_rows:\r\n'
    b'        raise SystemExit("No tasks finished by every run yet. Use --all-tasks or finish the runs.")\r\n\r\n'
    b'    cols = list(out_rows[0].keys())'
)

assert target2 in raw, "target2 not found in summarize.py"
raw = raw.replace(target2, replacement2, 1)

path.write_bytes(raw)
print("Successfully patched summarize.py!")
