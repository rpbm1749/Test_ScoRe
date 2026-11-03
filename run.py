"""Run one experimental condition over a list of AppWorld tasks.

Examples
  python run.py --mode frontier --run-name frontier_only
  python run.py --mode executor --run-name executor_only
  python run.py --mode score --static-k --k 3 --run-name score_static_k3     # fixed k baseline
  python run.py --mode score --static-k --k 8 --run-name score_static_k8     # big fixed budget
  python run.py --mode score --k 3 --run-name score_dynamic                  # planner declares k per window
  python run.py --mode score --run-name score_dynamic --limit 5

Results go to runs/<run-name>/: one JSON per task (full trajectory) plus results.jsonl
(one summary line per task). Re-running the same command skips finished tasks, so if a
daily quota runs out you just run it again the next day.
"""

from __future__ import annotations

import argparse
import os
import json
import sys
import time
import traceback
from pathlib import Path

import yaml
from dotenv import load_dotenv

# On Windows, Python writes files in a legacy encoding by default and AppWorld's evaluator
# crashes on non-ASCII text ("'charmap' codec can't encode"). Re-launch in UTF-8 mode.
if os.name == "nt" and not sys.flags.utf8_mode:
    import subprocess

    sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from score.agents import run_score, run_single  # noqa: E402
from score.llm import QuotaExhausted, RequestTooLarge, build_client  # noqa: E402


def load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_task_list(cfg: dict, args: argparse.Namespace) -> list[str]:
    if args.task_ids:
        return args.task_ids
    path = ROOT / (args.tasks or cfg["tasks"]["file"])
    if not path.exists():
        raise SystemExit(f"Task list {path} not found. Run `python select_tasks.py` first.")
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]
    return ids[: args.limit] if args.limit else ids


def done_task_ids(results_path: Path) -> set[str]:
    """Tasks already finished. Tasks whose latest attempt crashed (end_reason "error") or whose
    grading failed (eval_error) are retried;
    summarize.py uses the latest line per task, so the retry replaces the crashed attempt."""
    if not results_path.exists():
        return set()
    latest: dict[str, dict] = {}
    for line in results_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            latest[row["task_id"]] = row
    return {t for t, row in latest.items() if row.get("end_reason") != "error" and not row.get("eval_error")}


def make_env(task_id: str, run_name: str, max_steps: int):
    from score.env import AppWorldEnv

    return AppWorldEnv(task_id=task_id, experiment_name=f"score_{run_name}", max_steps=max_steps)


def window_stats(windows: list[dict]) -> dict:
    """Per-task dynamic-k statistics from the logged windows."""
    n = len(windows)
    if not n:
        return {"windows": 0, "avg_granted_k": 0, "avg_used_steps": 0, "pct_declared": 0,
                "pct_stale": 0, "budget_exhausted_windows": 0}
    granted = [w.get("granted_k") or 0 for w in windows]
    used = [len(w.get("executor_steps") or []) for w in windows]
    return {
        "windows": n,
        "avg_granted_k": round(sum(granted) / n, 2),
        "avg_used_steps": round(sum(used) / n, 2),
        "pct_declared": round(sum(1 for w in windows if w.get("k_source") == "declared") / n, 3),
        "pct_stale": round(sum(1 for w in windows if w.get("stale_instructions")) / n, 3),
        "budget_exhausted_windows": sum(1 for w in windows if str(w.get("handback_reason", "")).startswith("budget_exhausted")),
    }


def summarise(task_id: str, mode: str, k: int | None, end_reason: str, traj_d: dict, evaluation: dict, error: str | None) -> dict:
    usage = traj_d["usage"]
    frontier_role = "planner" if mode == "score" else ("frontier" if mode == "frontier" else None)
    cheap_role = "executor" if mode == "score" else ("executor" if mode == "executor" else None)
    fu = usage.get(frontier_role, {}) if frontier_role else {}
    cu = usage.get(cheap_role, {}) if cheap_role else {}
    row = {
        "task_id": task_id,
        "mode": mode,
        "k": k,
        "planner_view": traj_d.get("planner_view"),
        "dynamic_k": traj_d.get("dynamic_k"),
        "success": evaluation.get("success", False),
        "pass_percentage": evaluation.get("pass_percentage", 0.0),
        "eval_error": evaluation.get("eval_error"),
        "end_reason": end_reason,
        "error": error,
        "env_steps": sum(traj_d["steps_by_actor"].values()),
        "steps_by_actor": traj_d["steps_by_actor"],
        "wall_time_s": traj_d["wall_time_s"],
        "fetches": sum(1 for c in traj_d.get("calls", []) if str(c.get("note", "")).startswith("fetch:")),
        "rejections": len(traj_d.get("rejections", [])),
        **window_stats(traj_d.get("windows", [])),
    }
    for prefix, u in (("frontier", fu), ("cheap", cu)):
        for key in (
            "calls", "estimated_calls", "trimmed_calls", "prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens",
            "latency_s", "queue_time_s", "prompt_time_s", "completion_time_s", "total_time_s",
        ):
            row[f"{prefix}_{key}"] = u.get(key, 0)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["frontier", "executor", "score"])
    ap.add_argument("--run-name", required=True, help="folder name under runs/")
    ap.add_argument("--k", type=int, help="score mode: k_default (the fixed k with --static-k, else the fallback when the planner declares none)")
    ap.add_argument("--static-k", action="store_true", help="score mode: ignore the planner's EXECUTOR_STEPS and always grant k_default")
    ap.add_argument("--view", choices=["notes", "full"], help="planner view (score mode); overrides config")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--tasks", help="task list file (default: tasks.file in config)")
    ap.add_argument("--task-ids", nargs="*", help="run these task IDs instead of a file")
    ap.add_argument("--limit", type=int, help="only the first N tasks of the list")
    ap.add_argument("--allow-config-change", action="store_true", help="resume a run even though the config differs from its saved config_used.yaml")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = load_config(ROOT / args.config)
    if args.k is not None:
        cfg["score"]["k_default"] = args.k   # agents.py reads k_default first
        cfg["score"].pop("k", None)
    if args.view is not None:
        cfg["score"]["planner_view"] = args.view
    if args.static_k:
        cfg["score"]["dynamic_k"] = False
    verbose = not args.quiet

    k = int(cfg["score"].get("k_default", cfg["score"].get("k", 3))) if args.mode == "score" else None
    dynamic_k = bool(cfg["score"].get("dynamic_k", True)) if args.mode == "score" else None
    planner = executor = None
    if args.mode in ("frontier", "score"):
        planner = build_client("planner" if args.mode == "score" else "frontier", cfg["models"]["planner"], cfg["providers"], verbose)
    if args.mode in ("executor", "score"):
        executor = build_client("executor", cfg["models"]["executor"], cfg["providers"], verbose)

    out_dir = ROOT / "runs" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    cfg_path = out_dir / "config_used.yaml"
    cfg_now = {**cfg, "mode": args.mode}
    if cfg_path.exists():
        saved = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        if saved != cfg_now and not args.allow_config_change:
            raise SystemExit(
                f"Config for run '{args.run_name}' differs from its saved config_used.yaml, so resuming would mix "
                f"conditions in one run. Use a new --run-name, or pass --allow-config-change to continue anyway."
            )
        if saved != cfg_now:
            print("WARNING: config differs from the saved one; continuing because --allow-config-change was given.")
    else:
        cfg_path.write_text(yaml.safe_dump(cfg_now, sort_keys=False), encoding="utf-8")

    task_ids = load_task_list(cfg, args)
    done = done_task_ids(results_path)
    todo = [t for t in task_ids if t not in done]
    print(f"Run '{args.run_name}' | mode={args.mode} k={k} dynamic_k={dynamic_k} | {len(done)} done, {len(todo)} to go")

    for n, task_id in enumerate(todo, 1):
        print(f"\n[{n}/{len(todo)}] task {task_id}")
        env = make_env(task_id, args.run_name, cfg["run"]["max_steps"])
        if verbose:
            print(f"  Task: {env.instruction}")
        error = None
        extra: dict = {}
        try:
            if args.mode == "frontier":
                end_reason, traj, extra = run_single(env, planner, cfg, role="frontier", verbose=verbose)
            elif args.mode == "executor":
                end_reason, traj, extra = run_single(env, executor, cfg, role="executor", verbose=verbose)
            else:
                end_reason, traj, extra = run_score(env, planner, executor, cfg, verbose=verbose)
        except QuotaExhausted as e:
            env.close()
            print(f"\nDaily quota exhausted: {e}\nTask {task_id} was not recorded. Re-run the same command tomorrow to continue.")
            sys.exit(2)
        except RequestTooLarge as e:
            # Record it: the request outgrew the per-minute token limit. That is a real result on the free tier.
            error = f"request_too_large: {e}"
            end_reason, traj = "request_too_large", getattr(e, "traj", None)
        except Exception as e:  # keep going on unexpected errors, but record them
            error = f"{e.__class__.__name__}: {e}"
            end_reason, traj = "error", getattr(e, "traj", None)
            traceback.print_exc()

        if traj is None:
            from score.state import Trajectory

            traj = Trajectory(task_id, args.mode, k)
        try:
            evaluation = env.evaluate()
        except Exception as e:
            evaluation = {"success": False, "pass_percentage": 0.0, "eval_error": str(e)}
        env.close()

        traj_d = traj.to_dict()
        traj_d.update({"end_reason": end_reason, "error": error, "evaluation": evaluation, **extra,
                       "planner_view": cfg["score"].get("planner_view", "full") if args.mode == "score" else None,
                       "dynamic_k": dynamic_k})
        (out_dir / f"{task_id}.json").write_text(json.dumps(traj_d, indent=1), encoding="utf-8")
        row = summarise(task_id, args.mode, k, end_reason, traj_d, evaluation, error)
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(
            f"  -> {'SUCCESS' if row['success'] else 'fail'} ({row['pass_percentage']}% tests) | end={end_reason} | "
            f"steps={row['env_steps']} {row['steps_by_actor']} | frontier calls={row['frontier_calls']} "
            f"tokens={row['frontier_prompt_tokens'] + row['frontier_completion_tokens']} | "
            f"cheap calls={row['cheap_calls']} tokens={row['cheap_prompt_tokens'] + row['cheap_completion_tokens']}"
        )
        if args.mode == "score":
            print(f"     windows={row['windows']} avg_granted_k={row['avg_granted_k']} avg_used={row['avg_used_steps']} "
                  f"declared={row['pct_declared']:.0%} stale={row['pct_stale']:.0%}")
        time.sleep(0.5)

    print("\nDone. Summarise with: python summarize.py")


if __name__ == "__main__":
    main()