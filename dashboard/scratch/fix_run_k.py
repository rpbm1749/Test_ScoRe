from pathlib import Path

p = Path("d:/projects/Test_ScoRe/run.py")
raw = p.read_bytes()

OLD = b"k = int(cfg[\"score\"][\"k\"]) if args.mode == \"score\" else None"
NEW = b"k = int(cfg[\"score\"].get(\"k\", cfg[\"score\"].get(\"k_default\", 3))) if args.mode == \"score\" else None"

assert OLD in raw, "Line in run.py not found!"
raw = raw.replace(OLD, NEW)
p.write_bytes(raw)
print("Updated run.py with defensive fallback for k.")
