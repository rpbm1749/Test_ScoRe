"""Check the setup before spending quota on real runs.

  python smoke_test.py            # API key, both models, AppWorld data, one env step
  python smoke_test.py --no-env   # skip the AppWorld check
"""

from __future__ import annotations

import argparse
import os
import sys
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

from score.actions import TOOLS, parse_action  # noqa: E402
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
            res = client.chat(
                [{"role": "user", "content": "Call the execute_python tool with the code: print('ok')"}], tools=TOOLS
            )
            act = parse_action(res)
            how = "tool call" if act.via_tool else ("recovered/fallback" if act.code else "NO CODE")
            print(f"OK ({res.usage.prompt_tokens}+{res.usage.completion_tokens} tokens, {how}): {act.code!r}")
        except SystemExit as e:
            ok = False
            print(f"FAILED: {e}")
        except Exception as e:
            ok = False
            print(f"FAILED: {e.__class__.__name__}: {e}")
            if mcfg["provider"] == "nvidia":
                print("   Check NVIDIA_API_KEY in .env (from build.nvidia.com). Some models need you to open")
                print("   their page on build.nvidia.com once and accept terms before the API works.")
            if mcfg["provider"] == "ollama":
                print("   Is Ollama running? Try `ollama run gpt-oss:20b` in another terminal.")

    if not args.no_env:
        print("[appworld] loading task list and running one step ... ", end="", flush=True)
        try:
            from appworld import load_task_ids

            from score.env import AppWorldEnv

            ids = load_task_ids("dev", difficulty=3)
            env = AppWorldEnv(task_id=ids[0], experiment_name="smoke_test", max_steps=5)
            try:
                out = env.execute("print(apis.api_docs.show_app_descriptions())")
            finally:
                env.close()
            print(f"OK ({len(ids)} hard dev tasks; first output {len(out)} chars)")
        except Exception as e:
            ok = False
            print(f"FAILED: {e.__class__.__name__}: {e}")
            print("   Did you run `appworld install` and `appworld download data` in this folder?")

    print("\nAll good." if ok else "\nFix the failures above before running experiments.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
