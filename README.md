# Test_ScoRe

> **Team members: start with [TEAM_GUIDE.md](TEAM_GUIDE.md)** for setup, the experiments to run, and how to share results.

Testing the feasibility of ScoRe: a frontier **planner** takes one step and writes instructions,
then a cheaper **executor** takes the next *k* steps, then control goes back to the planner.

- Task suite: [AppWorld](https://github.com/StonyBrookNLP/appworld): everyday tasks across simulated apps (email, Spotify, Venmo, file system...), solved by writing Python against the apps' APIs. Hard (difficulty 3) tasks typically take 20+ steps, and AppWorld checks the final state automatically.
- Executor / cheap: `openai/gpt-oss-20b`. Planner / frontier: `nvidia/nemotron-3-super-120b-a12b` (120B total, 12B active), because NVIDIA retired gpt-oss-120b on 2026-09-03. Both run on NVIDIA's free API (build.nvidia.com: no payment details, 40 requests/min shared, no token caps listed).
- NVIDIA's model list changes often. `python probe_models.py` lists what's live, tests candidates on a two-turn tool task, and `--use <model-id>` sets the planner in `config.yaml`.
- Groq's free tier also works (`config.yaml` explains how), but it rejects any request over ~8K tokens, so long tasks need their history trimmed there.

## Setup (once)

Needs **Python 3.11 or newer** (AppWorld requirement). In PowerShell, from this folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

appworld install
appworld download data          # downloads the tasks into .\data (a few hundred MB)

copy .env.example .env          # then open .env and paste your NVIDIA key
python -m pytest tests -q       # offline tests of the loop, no key or data needed
python probe_models.py          # which frontier models are live; then: python probe_models.py --use <model-id>
python smoke_test.py            # checks the key, both models and AppWorld
python select_tasks.py          # writes tasks\dev_hard.txt (20 hard dev tasks)
```

Key: sign up at build.nvidia.com (NVIDIA developer account, no card), then create an API key (starts with `nvapi-`). If a model returns an access error, open that model's page on build.nvidia.com once.

If `pip install` or `appworld install` fails on Windows, run the same steps inside WSL (Ubuntu), using `python3.11 -m venv .venv` and `source .venv/bin/activate`.

## Running experiments

Each command runs one condition over the task list. Re-running the same command skips tasks that are already finished, so when the daily quota runs out (the script stops and says so), just run it again the next day.

```powershell
python run.py --mode executor --run-name executor_only     # 20B alone: lower reference
python run.py --mode frontier --run-name frontier_only     # 120B alone: upper reference
python run.py --mode score --k 3 --run-name score_k3       # ScoRe, k = 3
python run.py --mode score --k 5 --run-name score_k5       # ...change k per run
python run.py --mode score --k 3 --view full --run-name score_k3_full   # the growing-history planner, for comparison
python summarize.py                                        # comparison table -> runs\summary.csv
```

Handy flags: `--limit 3` (first 3 tasks only, good for a first try), `--task-ids <id> <id>`, `--quiet`.

**Start small:** run `--limit 2` for each mode first and look at the token counts printed per task. That tells you how many tasks per day the free quota allows.

## What gets recorded

`runs/<run-name>/`
- `results.jsonl`: one line per task: success, % of AppWorld's checks passed, why it ended, steps by actor, wall time, and for each model (frontier / cheap): calls, prompt / completion / cached / reasoning tokens, our measured latency, and the provider's reported queue / prompt / completion / total time when it sends them (Groq and Cerebras do; others may report 0).
- `<task_id>.json`: full trajectory: every step's code, raw output and truncated output, every LLM call with its usage, and (for ScoRe) every window's instruction block, executor steps and handback reason.
- `config_used.yaml`: the exact config of that run.

`summarize.py` compares runs on the tasks they have all finished: success rate, average steps, frontier calls, tokens, reasoning tokens, provider-reported hardware time (prompt + completion) and queue time, estimated compute (TFLOPs ≈ 2 × active parameters × uncached tokens; 5.1B active for 120B, 3.6B for 20B), estimated USD at Groq's paid prices, and **modelled relative cost** for frontier/cheap price ratios R = 2, 5, 10, 20. At Groq's prices the 120B costs exactly 2x the 20B, so real dollars show only a small effect. The main metrics are frontier calls removed and success rate, with cost shown as "what if the frontier model were R times pricier".

## How the orchestration works

```
planner turn  ->  one step (code) + <instructions> block
executor window -> up to k steps, following the instructions
handback       ->  planner sees its step's output + all executor steps + the reason
...repeat until apis.supervisor.complete_task() is called or max_steps is reached
```

- **Global state (planner):** the planner's own running conversation: task, every instruction block it wrote, and every step by either model, with outputs cut by the same rule as the baselines (`run.max_output_chars`). It sees the same information a single-model agent would, so differences in success come from delegation, not lost information. Because it only grows by appending, it is also friendly to prompt caching.
- **Local state (executor):** rebuilt for every window: the task, the planner's latest instruction block, the planner's latest step and output, and only the executor's own steps in this window.
- **Planner view (`score.planner_view`, or `--view`):**
  - `notes` (default): each planner call is built fresh from the planner's own **progress notes** (written by the planner every turn, as a third argument of its tool call), a **one-line log** of every step (number, who, ok/ERROR, first line of code), and the **full outputs of only the steps since its last turn**. Its prompt stays roughly constant instead of growing with the task. Older outputs can be re-read with a `get_step_output(step)` tool, up to `max_fetches_per_turn` times per turn; every fetch is logged (`fetches` in the results).
  - `full`: the first version. The planner keeps one growing conversation containing every step's full output.
- **Context the orchestrator adds (`score/context.py`, task-agnostic):** the executor's conversation is wiped every window, but the environment's state and anything already looked up persist. So every executor window (and, in the notes view, every planner turn) also gets:
  - **Current state** from the environment's `state_summary()` hook. For AppWorld: the variables in the Python session, with type and a short preview.
  - **Reference** collected by the environment's `reference_from_step(code, output)` hook after every step. For AppWorld: one-line API signatures and app/API lists from every documentation lookup so far.
  - **The planner's progress notes** (notes view).
  
  These are built by code, not written by a model. A new environment only needs to implement the two hooks (both optional) in its adapter; the AppWorld versions live in `score/env.py`.
- **Pre-execution check (all modes, including both baselines):** before any step runs, the environment's `validate(code)` hook can stop it and send the model a specific correction instead; the step isn't run and doesn't use an environment step. The AppWorld version checks against the task's own API catalog (`task.api_docs`) and the session's real variables: syntax errors, apps/APIs that don't exist, parameters an API doesn't have (AppWorld would silently ignore them and return wrong data), missing required parameters, doc lookups for APIs that don't exist, and undefined variables. After `run.max_rejections_in_a_row` rejections the code runs anyway, so the check can never block a run. Every rejection is logged (`rejections`). Replayed on the first five runs: 0 false alarms in 128 working steps, 7 of 10 failing steps caught, plus 6 steps that "worked" but silently used a wrong parameter.
- **Automatic API reference:** every API a step calls or looks up, and every `app.api` the planner's instructions name, is added to the reference card straight from the task's catalog, so the executor has the real signature before its first step. In the notes view the planner sees the card too.
- **Prompt rules that apply to any task:** the executor prints the raw records behind any fact it extracts (never only its conclusion), hands back as soon as the success check is met, and doesn't re-look-up what's in the reference. The planner names exact APIs and variables, and confirms key facts from raw data before irreversible actions.
- **Handback reasons:** `budget_exhausted (k=…)`, `executor_handback: <reason>` (the executor replied `HANDBACK: …`), `consecutive_errors`, `executor_ready_to_finish`, `step_budget_reserved_for_planner`.
- **The planner always takes the final step.** The executor is not allowed to call `apis.supervisor.complete_task`: if it tries, that code is not run but passed to the planner as a proposal, and the planner checks the work and finishes the task itself. The last step of the `max_steps` budget is also kept for the planner, and it is told explicitly when it is on its last step.
- Both models share one Python REPL, so variables (e.g. access tokens) carry across windows. The planner is told to list the ones the executor needs under `KNOWN:`.

Models run code by calling an `execute_python` tool, one call per step. gpt-oss is trained to run code through a tool, and without one declared the API rejects its replies. A ```python block in the reply text is accepted as a fallback. If the provider can't parse a tool call (gpt-oss sometimes calls its built-in `python` tool with raw code), the code is recovered from the error, and that call's token counts are estimated and flagged (`estimated_calls`).

Models may not reassign `apis` (the object that reaches every app). The prompt says so, and if a step does it anyway, the original is restored and the output says so. This applies to every mode.

Every mode gets the same AppWorld guide prompt (adapted from AppWorld's official ReAct prompt), the same step budget and the same output truncation.

## Config (`config.yaml`)

- `score.k`: executor steps per window (or pass `--k`).
- `score.allow_early_handback`: let the executor hand back before k steps. Set `false` for a strict fixed-k experiment.
- `score.handback_on_consecutive_errors`: `0` disables it.
- `models.*.max_context_tokens` (optional, off by default): needed only on providers that cap request size, like Groq's free tier. When a model's history is longer than this (conservative estimate), the oldest steps are left out of what's sent. The task prompt and the newest steps are always kept, a note tells the model earlier steps were removed, and the full history is still logged. Trimmed calls are counted (`trimmed_calls`); if you use it, report it as a limitation.
- `models.*.reasoning_effort`: `low` by default to save quota. Keep it the same across runs you compare.
- To run the executor locally: install [Ollama](https://ollama.com), run `ollama pull gpt-oss:20b`, then set `executor.provider: ollama` and `executor.model: gpt-oss:20b` (and remove its `rpm`/`tpm`). gpt-oss-20b needs roughly 16 GB of RAM/VRAM and will be slow on CPU.

## Free-tier notes

NVIDIA: 40 requests/min shared across all models (the two clients use 20 each). Groq, if you switch to it: 30 requests/min, 8K tokens/min and 200K tokens/day per model. The client throttles itself under the per-minute limits and retries on 429s; on an hourly limit it waits, and on a daily limit `run.py` stops without recording the unfinished task, so run it again later. A request too large for the provider is recorded as `end_reason: request_too_large`.

## Files

| File | What it is |
|---|---|
| `run.py` | Runs one condition over the task list |
| `summarize.py` | Compares runs, writes `runs/summary.csv` |
| `select_tasks.py` | Picks the task set |
| `smoke_test.py` | Checks key, models and AppWorld before spending quota |
| `probe_models.py` | Lists live NVIDIA models, tests them on a tool task, sets the planner model |
| `config.yaml` | Models, k, step budget, truncation |
| `score/agents.py` | The loops: single-model ReAct and ScoRe planner/executor |
| `score/state.py` | Global state, local state, trajectory log |
| `score/context.py` | Task-agnostic state + reference context for the models (environment hooks) |
| `score/prompts.py` | All prompts |
| `score/llm.py` | API client (NVIDIA, Groq, Cerebras, Ollama) with throttling, retries, optional history trimming and usage tracking |
| `score/env.py` | AppWorld wrapper |
| `score/parsing.py` | Extracts code, instruction blocks and HANDBACK lines; repairs malformed tool calls |
| `score/actions.py` | The `execute_python` tool, and turning replies into steps |
| `tests/` | Offline tests with a fake environment and scripted models |
