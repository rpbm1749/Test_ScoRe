"""Check the setup before spending quota on real runs.

  python smoke_test.py            # API key, both models, AppWorld data, one env step
  python smoke_test.py --no-env   # skip the AppWorld check
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from score.llm import build_client  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-env", action="store_true")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    ok = True

    for role in ("planner", "executor"):
        mcfg = cfg["models"][role]
        print(f"[{role}] {mcfg['provider']} / {mcfg['model']} ... ", end="", flush=True)
        try:
            client = build_client(role, mcfg, cfg["providers"], verbose=False)
            res = client.chat([{"role": "user", "content": "Reply with exactly: ```python\nprint('ok')\n```"}])
            print(f"OK ({res.usage.prompt_tokens}+{res.usage.completion_tokens} tokens): {res.content.strip()[:60]!r}")
        except SystemExit as e:
            ok = False
            print(f"FAILED: {e}")
        except Exception as e:
            ok = False
            print(f"FAILED: {e.__class__.__name__}: {e}")
            if mcfg["provider"] == "ollama":
                print("   Is Ollama running? Try `ollama run gpt-oss:20b` in another terminal.")

    if not args.no_env:
        print("[appworld] loading task list and running one step ... ", end="", flush=True)
        try:
            from appworld import AppWorld, load_task_ids

            ids = load_task_ids("dev", difficulty=3)
            with AppWorld(task_id=ids[0], experiment_name="smoke_test") as world:
                out = world.execute("print(apis.api_docs.show_app_descriptions())")
            print(f"OK ({len(ids)} hard dev tasks; first output {len(out)} chars)")
        except Exception as e:
            ok = False
            print(f"FAILED: {e.__class__.__name__}: {e}")
            print("   Did you run `appworld install` and `appworld download data` in this folder?")

    print("\nAll good." if ok else "\nFix the failures above before running experiments.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
