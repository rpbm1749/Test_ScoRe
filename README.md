# Test_ScoRe

Testing the feasibility of ScoRe: a frontier **planner** takes one step and writes instructions,
then a cheaper **executor** takes the next *k* steps, then control goes back to the planner.

- Task suite: [AppWorld](https://github.com/StonyBrookNLP/appworld): everyday tasks across simulated apps (email, Spotify, Venmo, file system...), solved by writing Python against the apps' APIs. Hard (difficulty 3) tasks typically take 20+ steps, and AppWorld checks the final state automatically.
- Planner / frontier: `openai/gpt-oss-120b` on Groq (free tier).
- Executor / cheap: `openai/gpt-oss-20b` on Groq, or locally through Ollama.

## Setup (once)

Needs **Python 3.11 or newer** (AppWorld requirement). In PowerShell, from this folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

appworld install
appworld download data          # downloads the tasks into .\data (a few hundred MB)

copy .env.example .env          # then open .env and paste your Groq key
python -m pytest tests -q       # offline tests of the loop, no key or data needed
python smoke_test.py            # checks the key, both models and AppWorld
python select_tasks.py          # writes tasks\dev_hard.txt (20 hard dev tasks)
```

If `pip install` or `appworld install` fails on Windows, run the same steps inside WSL (Ubuntu), using `python3.11 -m venv .venv` and `source .venv/bin/activate`.

## Running experiments

Each command runs one condition over the task list. Re-running the same command skips tasks that are already finished, so when the daily quota runs out (the script stops and says so), just run it again the next day.

```powershell
python run.py --mode executor --run-name executor_only     # 20B alone: lower reference
python run.py --mode frontier --run-name frontier_only     # 120B alone: upper reference
python run.py --mode score --k 3 --run-name score_k3       # ScoRe, k = 3
python run.py --mode score --k 5 --run-name score_k5       # ...change k per run
python summarize.py                                        # comparison table -> runs\summary.csv
```

Handy flags: `--limit 3` (first 3 tasks only, good for a first try), `--task-ids <id> <id>`, `--quiet`.

**Start small:** run `--limit 2` for each mode first and look at the token counts printed per task. That tells you how many tasks per day the free quota allows.

## What gets recorded

`runs/<run-name>/`
- `results.jsonl`: one line per task: success, % of AppWorld's checks passed, why it ended, steps by actor, frontier and cheap calls and tokens (prompt / completion / cached), wall time.
- `<task_id>.json`: full trajectory: every step's code, raw output and truncated output, every LLM call with its usage, and (for ScoRe) every window's instruction block, executor steps and handback reason.
- `config_used.yaml`: the exact config of that run.

`summarize.py` compares runs on the tasks they have all finished: success rate, average steps, frontier calls, tokens, and **modelled relative cost** for frontier/cheap price ratios R = 1, 5, 10, 20. Both gpt-oss models cost about the same per token, so real dollars would hide the effect. The main metrics are frontier calls removed and success rate, with cost shown as "what if the frontier model were R times pricier".

## How the orchestration works

```
planner turn  ->  one step (code) + <instructions> block
executor window -> up to k steps, following the instructions
handback       ->  planner sees its step's output + all executor steps + the reason
...repeat until apis.supervisor.complete_task() is called or max_steps is reached
```

- **Global state (planner):** the planner's own running conversation: task, every instruction block it wrote, and every step by either model, with outputs cut by the same rule as the baselines (`run.max_output_chars`). It sees the same information a single-model agent would, so differences in success come from delegation, not lost information. Because it only grows by appending, it is also friendly to prompt caching.
- **Local state (executor):** rebuilt for every window: the task, the planner's latest instruction block, the planner's latest step and output, and only the executor's own steps in this window.
- **Handback reasons:** `budget_exhausted (k=…)`, `executor_handback: <reason>` (the executor replied `HANDBACK: …`), `consecutive_errors`, `task_completed`, `max_steps`.
- Both models share one Python REPL, so variables (e.g. access tokens) carry across windows. The planner is told to list the ones the executor needs under `KNOWN:`.

Every mode gets the same AppWorld guide prompt (adapted from AppWorld's official ReAct prompt), the same step budget and the same output truncation.

## Config (`config.yaml`)

- `score.k`: executor steps per window (or pass `--k`).
- `score.allow_early_handback`: let the executor hand back before k steps. Set `false` for a strict fixed-k experiment.
- `score.handback_on_consecutive_errors`: `0` disables it.
- `models.*.reasoning_effort`: `low` by default to save quota. Keep it the same across runs you compare.
- To run the executor locally: install [Ollama](https://ollama.com), run `ollama pull gpt-oss:20b`, then set `executor.provider: ollama` and `executor.model: gpt-oss:20b` (and remove its `rpm`/`tpm`). gpt-oss-20b needs roughly 16 GB of RAM/VRAM and will be slow on CPU.

## Free-tier notes (Groq)

The free tier allows about 30 requests/min, 8K tokens/min and 200K tokens/day **per model**. The client throttles itself under the per-minute limits and retries on 429s.
- If a daily limit is hit, `run.py` stops without recording the unfinished task. Run it again later.
- If a single request is larger than the per-minute token limit, the task is recorded with `end_reason: request_too_large`. Groq says cached tokens don't count toward limits, which should help long conversations, but check this on your first runs. If it happens a lot in frontier-only runs, that is a real limitation of the free setup, and worth reporting.

## Files

| File | What it is |
|---|---|
| `run.py` | Runs one condition over the task list |
| `summarize.py` | Compares runs, writes `runs/summary.csv` |
| `select_tasks.py` | Picks the task set |
| `smoke_test.py` | Checks key, models and AppWorld before spending quota |
| `config.yaml` | Models, k, step budget, truncation |
| `score/agents.py` | The loops: single-model ReAct and ScoRe planner/executor |
| `score/state.py` | Global state, local state, trajectory log |
| `score/prompts.py` | All prompts |
| `score/llm.py` | Groq/Ollama client with throttling, retries and usage tracking |
| `score/env.py` | AppWorld wrapper |
| `score/parsing.py` | Extracts code, instruction blocks and HANDBACK lines |
| `tests/` | Offline tests with a fake environment and scripted models |
