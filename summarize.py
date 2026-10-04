"""Compare all runs under runs/ and write runs/summary.csv.

Only tasks that every compared run has finished are used for the comparison, so a run
that is half-done (e.g. waiting on tomorrow's quota) doesn't skew the numbers.

Two kinds of cost are reported:
  * est_usd_per_task: what the run would cost at Groq's paid prices (price_per_m in
    config.yaml), with cached input tokens billed at the cached rate. On Groq the 120B
    model is exactly 2x the 20B model.
  * avg_est_tflops / rel_tflops: compute estimate, 2 x active parameters x tokens processed.
  * avg_groq_hw_s / rel_groq_hw_time: hardware time Groq reports for each request.
  * rel_cost_R<r>: total tokens priced as if the frontier model cost r times the cheap
    model, relative to the frontier-only run. This shows what ScoRe would save with a
    bigger price gap (e.g. a real frontier model vs a small one).

  python summarize.py
  python summarize.py --runs frontier_only executor_only score_k3 score_k5
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
RATIOS = [2, 5, 10, 20]


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

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    fp = cfg["models"]["planner"].get("price_per_m")
    cp = cfg["models"]["executor"].get("price_per_m")

    def usd(r: dict) -> float | None:
        if not fp or not cp:
            return None
        total = 0.0
        for prefix, price in (("frontier", fp), ("cheap", cp)):
            cached = r.get(f"{prefix}_cached_tokens", 0)
            uncached = r[f"{prefix}_prompt_tokens"] - cached
            total += uncached * price["input"] + cached * price["cached_input"] + r[f"{prefix}_completion_tokens"] * price["output"]
        return total / 1e6

    fa = cfg["models"]["planner"].get("active_params_b")
    ca = cfg["models"]["executor"].get("active_params_b")

    def tflops(r: dict) -> float | None:
        """~2 x active params x tokens processed. Cached prompt tokens are skipped (the provider
        reuses earlier work for them). Ignores attention cost, so it is a lower-bound estimate."""
        if not fa or not ca:
            return None
        total = 0.0
        for prefix, active_b in (("frontier", fa), ("cheap", ca)):
            processed = r[f"{prefix}_prompt_tokens"] - r.get(f"{prefix}_cached_tokens", 0) + r[f"{prefix}_completion_tokens"]
            total += 2 * active_b * 1e9 * processed
        return total / 1e12

    def hw_time(r: dict) -> float:
        """Groq-reported hardware time (prompt + completion), both models."""
        return sum(r.get(f"{p}_prompt_time_s", 0) + r.get(f"{p}_completion_time_s", 0) for p in ("frontier", "cheap"))

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
            "view": rows[0].get("planner_view") or "",
            "tasks": n,
            "success_rate": round(sum(r["success"] for r in rows) / n, 3),
            "avg_pass_pct": round(sum(r["pass_percentage"] for r in rows) / n, 1),
            "avg_env_steps": round(sum(r["env_steps"] for r in rows) / n, 1),
            "avg_frontier_calls": round(sum(r["frontier_calls"] for r in rows) / n, 1),
            "avg_cheap_calls": round(sum(r["cheap_calls"] for r in rows) / n, 1),
            "avg_frontier_tokens": round(sum(r["frontier_prompt_tokens"] + r["frontier_completion_tokens"] for r in rows) / n),
            "avg_cheap_tokens": round(sum(r["cheap_prompt_tokens"] + r["cheap_completion_tokens"] for r in rows) / n),
            "avg_fetches": round(sum(r.get("fetches", 0) for r in rows) / n, 1),
            "avg_rejections": round(sum(r.get("rejections", 0) for r in rows) / n, 1),
            "errors": sum(1 for r in rows if r.get("error")),
            "estimated_usage_calls": sum(r.get("frontier_estimated_calls", 0) + r.get("cheap_estimated_calls", 0) for r in rows),
            "trimmed_calls": sum(r.get("frontier_trimmed_calls", 0) + r.get("cheap_trimmed_calls", 0) for r in rows),
        }
        row["avg_reasoning_tokens"] = round(sum(r.get("frontier_reasoning_tokens", 0) + r.get("cheap_reasoning_tokens", 0) for r in rows) / n)
        row["avg_groq_hw_s"] = round(sum(hw_time(r) for r in rows) / n, 3)
        row["avg_groq_queue_s"] = round(sum(r.get("frontier_queue_time_s", 0) + r.get("cheap_queue_time_s", 0) for r in rows) / n, 3)
        row["avg_latency_s"] = round(sum(r.get("frontier_latency_s", 0) + r.get("cheap_latency_s", 0) for r in rows) / n, 2)
        costs = [usd(r) for r in rows]
        if all(c is not None for c in costs):
            row["est_usd_per_task"] = round(sum(costs) / n, 5)
        fl = [tflops(r) for r in rows]
        if all(f is not None for f in fl):
            row["avg_est_tflops"] = round(sum(fl) / n, 2)
        if baseline and baseline in data:
            base_rows = [data[baseline][r["task_id"]] for r in rows if r["task_id"] in data[baseline]]
            for ratio in RATIOS:
                b = sum(cost(r, ratio) for r in base_rows)
                me = sum(cost(r, ratio) for r in rows if r["task_id"] in data[baseline])
                row[f"rel_cost_R{ratio}"] = round(me / b, 3) if b else None
            if fa and ca:
                bf = sum(tflops(r) for r in base_rows)
                mf = sum(tflops(r) for r in rows if r["task_id"] in data[baseline])
                row["rel_tflops"] = round(mf / bf, 3) if bf else None
            bh = sum(hw_time(r) for r in base_rows)
            mh = sum(hw_time(r) for r in rows if r["task_id"] in data[baseline])
            row["rel_groq_hw_time"] = round(mh / bh, 3) if bh else None
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
    print("est_usd_per_task: estimated cost per task at Groq's paid prices (price_per_m in config.yaml), cache discount included.")
    print("avg_est_tflops: ~2 x active params x uncached tokens, both models (lower bound; ignores attention).")
    print("avg_groq_hw_s: Groq-reported prompt + completion time, both models. avg_groq_queue_s: time waiting in Groq's queue.")
    print("avg_latency_s: time measured on our side waiting for responses (includes network and queue).")

    out = runs_dir / "summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out_rows)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
