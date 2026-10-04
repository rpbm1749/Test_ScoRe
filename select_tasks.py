"""Pick the task set for the experiments and write it to tasks/<name>.txt.

Default: AppWorld's dev split, difficulty 3 (the hardest, longest tasks), one task per
scenario (each AppWorld scenario has 3 near-identical variants, so this keeps the set
diverse), first 20 after a fixed shuffle.

  python select_tasks.py                     # -> tasks/dev_hard.txt, 20 tasks
  python select_tasks.py --n 30 --difficulty 2 --out tasks/dev_d2.txt
"""

from __future__ import annotations

import argparse
import sys
import os
import random
from pathlib import Path

# On Windows, Python writes files in a legacy encoding by default and AppWorld's evaluator
# crashes on non-ASCII text ("'charmap' codec can't encode"). Re-launch in UTF-8 mode.
if os.name == "nt" and not sys.flags.utf8_mode:
    import subprocess

    sys.exit(subprocess.call([sys.executable, "-X", "utf8", *sys.argv]))

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dev", choices=["train", "dev", "test_normal", "test_challenge"])
    ap.add_argument("--difficulty", type=int, default=3, choices=[1, 2, 3])
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--all-variants", action="store_true", help="keep all 3 variants of each scenario")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="tasks/dev_hard.txt")
    args = ap.parse_args()

    from appworld import load_task_ids

    ids = load_task_ids(
        args.dataset,
        difficulty=args.difficulty,
        num_tasks_per_scenario=None if args.all_variants else 1,
    )
    print(f"{len(ids)} tasks in {args.dataset} with difficulty {args.difficulty}")
    random.Random(args.seed).shuffle(ids)
    chosen = ids[: args.n]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    header = f"# dataset={args.dataset} difficulty={args.difficulty} n={len(chosen)} seed={args.seed}\n"
    out.write_text(header + "\n".join(chosen) + "\n", encoding="utf-8")
    print(f"Wrote {len(chosen)} task IDs to {out}")


if __name__ == "__main__":
    main()
