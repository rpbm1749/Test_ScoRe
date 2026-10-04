"""Find which large models NVIDIA's free API offers right now, and check they can run our agent loop.

NVIDIA retires models without much notice (gpt-oss-120b went away on 2026-09-03), so instead
of guessing we ask the API which models are live and test each candidate on a tiny two-turn
tool task: call execute_python, get the result back, answer.

  python probe_models.py                    # test the default candidates
  python probe_models.py --list             # just print every model the API offers
  python probe_models.py --models a/b c/d   # test specific model IDs
  python probe_models.py --use deepseek-ai/deepseek-v3.2   # set it as the planner in config.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from score.actions import TOOLS, observation_messages, parse_action  # noqa: E402
from score.llm import LLMClient  # noqa: E402

# Strong open models for agentic tool use, roughly in order of preference. Only the ones the
# API actually lists are tested.
CANDIDATES = [
    "moonshotai/kimi-k3",
    "deepseek-ai/deepseek-v4.1-flash",
    "deepseek-ai/deepseek-v4-pro",
    "deepseek-ai/deepseek-v4-flash",
    "z-ai/glm-5.3",
    "moonshotai/kimi-k2.6",
    "deepseek-ai/deepseek-v3.2",
    "z-ai/glm5",
    "qwen/qwen3-235b-a22b",
    "nvidia/llama-3.1-nemotron-ultra-253b-v1",
]
INTERESTING = re.compile(r"deepseek|kimi|glm|qwen3|nemotron-ultra|minimax|gpt-oss|inkling|llama-4", re.I)


def list_models(base_url: str, api_key: str) -> list[str]:
    r = httpx.get(f"{base_url}/models", headers={"Authorization": f"Bearer {api_key}"}, timeout=60)
    r.raise_for_status()
    return sorted(m["id"] for m in r.json().get("data", []))


def probe(base_url: str, api_key: str, model: str, timeout_s: float) -> dict:
    client = LLMClient(f"probe:{model}", base_url, model, api_key=api_key, max_tokens=2048,
                       pass_back_reasoning=True, max_retries=0, timeout_s=timeout_s, verbose=False)
    out = {"model": model, "tool_call": False, "answered": False, "latency_s": 0.0, "error": ""}
    try:
        msgs = [{"role": "user", "content": "Use the execute_python tool to compute 17*23 (print it), then tell me the number."}]
        r1 = client.chat(msgs, tools=TOOLS)
        act = parse_action(r1)
        out["latency_s"] = r1.usage.latency_s
        if not (act.code and act.via_tool):
            out["error"] = f"no tool call (reply: {r1.content[:80]!r})"
            return out
        out["tool_call"] = True
        msgs.append(act.assistant_message)
        msgs.extend(observation_messages(act, "391"))
        r2 = client.chat(msgs, tools=TOOLS)
        out["latency_s"] += r2.usage.latency_s
        out["answered"] = "391" in (r2.content or "")
        if not out["answered"]:
            out["error"] = f"second turn didn't answer (reply: {r2.content[:80]!r})"
    except Exception as e:  # report and move on
        msg = str(e)
        if "timeout" in msg.lower() or "timed out" in msg.lower():
            msg = f"no reply within {timeout_s:.0f}s (probably queued on the free tier)"
        out["error"] = f"{e.__class__.__name__}: {msg[:160]}"
    return out


def set_planner_model(model: str) -> None:
    path = ROOT / "config.yaml"
    text = path.read_text(encoding="utf-8")
    m = re.search(r"(\n  planner:\n(?:    .*\n|\s*#.*\n|\s*\n)*?    model: )(\S+)", text)
    if not m:
        raise SystemExit("Couldn't find planner.model in config.yaml; edit it by hand.")
    text = text[: m.start(2)] + model + text[m.end(2):]
    path.write_text(text, encoding="utf-8")
    print(f"config.yaml: planner.model is now {model}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="nvidia")
    ap.add_argument("--list", action="store_true", help="only list the models the API offers")
    ap.add_argument("--models", nargs="*", help="model IDs to test (default: built-in candidates)")
    ap.add_argument("--use", help="set this model as the planner in config.yaml and exit")
    ap.add_argument("--timeout", type=float, default=90, help="seconds to wait for each reply (default 90)")
    args = ap.parse_args()

    if args.use:
        set_planner_model(args.use)
        return

    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    pcfg = cfg["providers"][args.provider]
    api_key = os.environ.get(pcfg.get("api_key_env", ""), "")
    if not api_key:
        raise SystemExit(f"Set {pcfg.get('api_key_env')} in .env first.")
    base = pcfg["base_url"].rstrip("/")

    available = list_models(base, api_key)
    if args.list:
        print("\n".join(available))
        print(f"\n{len(available)} models")
        return
    print(f"{len(available)} models available. Large / relevant ones:")
    for mid in available:
        if INTERESTING.search(mid):
            print("  ", mid)

    wanted = args.models or CANDIDATES
    to_test = [m for m in wanted if m in available]
    missing = [m for m in wanted if m not in available]
    if missing:
        print(f"\nNot offered right now (skipped): {', '.join(missing)}")
    if not to_test:
        raise SystemExit("None of the candidates are offered. Pick from the list above and pass --models.")

    print(f"\nTesting {len(to_test)} model(s) on a two-turn tool task...\n")
    results = []
    for mid in to_test:
        print(f"  {mid} ... ", end="", flush=True)
        r = probe(base, api_key, mid, args.timeout)
        ok = r["tool_call"] and r["answered"]
        print(("OK" if ok else "FAIL") + f"  ({r['latency_s']:.1f}s)" + (f"  {r['error']}" if r["error"] else ""))
        results.append(r)
        time.sleep(2)  # stay well under the shared 40 requests/min

    good = [r for r in results if r["tool_call"] and r["answered"]]
    (ROOT / "runs").mkdir(exist_ok=True)
    (ROOT / "runs" / "probe_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    if good:
        best = good[0]["model"]
        print(f"\nWorking: {', '.join(r['model'] for r in good)}")
        print(f"Suggested planner: {best}   (set it with: python probe_models.py --use {best})")
        print("Prefer a fast one: a task makes dozens of planner calls, so a 30s+ model makes runs very slow.")
    else:
        print("\nNo candidate passed. Send the output above to Claude.")


if __name__ == "__main__":
    main()
