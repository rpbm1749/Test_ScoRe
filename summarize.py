"""Compare all runs under runs/ and write runs/summary.csv.

Only tasks that every compared run has finished are used for the comparison, so a run
that is half-done (e.g. waiting on tomorrow's quota) doesn't skew the numbers.

Cost is *modelled*: the two gpt-oss models cost about the same per token, so real dollars
would hide the effect. Instead we assume the frontier model costs R times the cheap model
per token and report total cost relative to the frontier-only run, for several R.

  python summarize.py
  python summarize.py --runs frontier_only executor_only score_k3 score_k5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RATIOS = [1, 5, 10, 20]


def load(run_dir: Path) -> dict[str, dict]:
    rows = {}
    path = run_dir / "results.jsonl"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows[r["task_id"]] = r
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", help="run names (default: all folders in runs/)")
    ap.add_argument("--baseline", default=None, help="run used as the cost reference (default: first run with mode=frontier)")
    ap.add_argument("--all-tasks", action="store_true", help="don't restrict to tasks every run has finished")
    args = ap.parse_args()

    runs_dir = ROOT / "runs"
    names = args.runs or sorted(p.name for p in runs_dir.iterdir() if (p / "results.jsonl").exists())
    data = {n: load(runs_dir / n) for n in names}
    data = {n: d for n, d in data.items() if d}
    if not data:
        raise SystemExit("No results yet. Run run.py first.")

    common = set.intersection(*(set(d) for d in data.values()))
    print(f"Runs: {', '.join(data)} | tasks finished by all: {len(common)}")

    def cost(r: dict, ratio: float) -> float:
        f = r["frontier_prompt_tokens"] + r["frontier_completion_tokens"]
        c = r["cheap_prompt_tokens"] + r["cheap_completion_tokens"]
        return f * ratio + c

    baseline = args.baseline or next((n for n, d in data.items() if next(iter(d.values()))["mode"] == "frontier"), None)

    out_rows = []
    for name, d in data.items():
        rows = list(d.values()) if args.all_tasks else [d[t] for t in sorted(common)]
        if not rows:
            continue
        n = len(rows)
        mode = rows[0]["mode"]
        row = {
            "run": name,
            "mode": mode,
            "k": rows[0].get("k"),
            "tasks": n,
            "success_rate": round(sum(r["success"] for r in rows) / n, 3),
            "avg_pass_pct": round(sum(r["pass_percentage"] for r in rows) / n, 1),
            "avg_env_steps": round(sum(r["env_steps"] for r in rows) / n, 1),
            "avg_frontier_calls": round(sum(r["frontier_calls"] for r in rows) / n, 1),
            "avg_cheap_calls": round(sum(r["cheap_calls"] for r in rows) / n, 1),
            "avg_frontier_tokens": round(sum(r["frontier_prompt_tokens"] + r["frontier_completion_tokens"] for r in rows) / n),
            "avg_cheap_tokens": round(sum(r["cheap_prompt_tokens"] + r["cheap_completion_tokens"] for r in rows) / n),
            "errors": sum(1 for r in rows if r.get("error")),
        }
        if baseline and baseline in data:
            base_rows = [data[baseline][r["task_id"]] for r in rows if r["task_id"] in data[baseline]]
            for ratio in RATIOS:
                b = sum(cost(r, ratio) for r in base_rows)
                me = sum(cost(r, ratio) for r in rows if r["task_id"] in data[baseline])
                row[f"rel_cost_R{ratio}"] = round(me / b, 3) if b else None
        out_rows.append(row)

    cols = list(out_rows[0].keys())
    for r in out_rows:
        for c in r:
            if c not in cols:
                cols.append(c)
    widths = {c: max(len(c), *(len(str(r.get(c, ""))) for r in out_rows)) for c in cols}
    print("  ".join(c.ljust(widths[c]) for c in cols))
    for r in out_rows:
        print("  ".join(str(r.get(c, "")).ljust(widths[c]) for c in cols))
    if baseline:
        print(f"\nrel_cost_R<r>: total cost relative to '{baseline}', if the frontier model costs r x the cheap model per token.")

    out = runs_dir / "summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
