# ScoRe — guide for the team

This is everything you need to clone the repo, set it up, run the experiments, and share results.
Read it once top to bottom before starting. The README has the design details.

## 1. What this project is (60 seconds)

ScoRe tests whether an AI agent can call an expensive **frontier model** only at checkpoints and
let a **cheap model** do the steps in between.

- **Planner (frontier):** `nvidia/nemotron-3-super-120b-a12b`. Takes one step, writes instructions
  and progress notes.
- **Executor (cheap):** `openai/gpt-oss-20b`. Takes up to *k* steps following the instructions,
  then hands control back.
- **Task suite:** [AppWorld](https://github.com/StonyBrookNLP/appworld): everyday tasks across
  simulated apps (Venmo, phone, Spotify, Gmail…), solved by writing Python against the apps' APIs,
  graded automatically.
- Both models run on **NVIDIA's free API** (no payment details).

We compare three conditions on the same tasks:

| Condition | Command flag | What it is |
|---|---|---|
| Executor only | `--mode executor` | the 20B alone (lower reference) |
| Frontier only | `--mode frontier` | Nemotron alone (upper reference) |
| ScoRe | `--mode score --k N` | planner + executor |

**Results so far (one task only: `530b157_1`, so not conclusive):**

| | Grade | Steps | Frontier calls | Frontier tokens |
|---|---|---|---|---|
| 20B alone | 20% | 11 | 0 | 0 |
| Nemotron alone | 100% | 19 | 19 | 140K |
| ScoRe k=3, full history to planner | 100% | 39 | 13 | 168K |
| ScoRe k=3, notes view (v1) | 90% | 33 | 10 | 61K |
| ScoRe k=3, notes view (v2) | 100% | 36 | 14 | 82K |

v3 (current code) adds a pre-execution checker, automatic API reference, and the planner seeing
the reference card. **Your job: run v3 on more tasks and the experiments in section 5.**

## 2. Setup (once per person)

You need **Python 3.11 or newer** and git.

### Windows (PowerShell)
```powershell
git clone <repo-url>
cd Test_ScoRe
py -3.11 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
appworld install
appworld download data
copy .env.example .env
```
If `activate` says scripts are disabled: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry.

### macOS / Linux
```bash
git clone <repo-url>
cd Test_ScoRe
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
appworld install
appworld download data
cp .env.example .env
```

### Your own NVIDIA API key
1. Sign up at **build.nvidia.com** (NVIDIA developer account; no card, but it may ask for phone verification).
2. Create an API key (starts with `nvapi-`).
3. Open `.env` and set `NVIDIA_API_KEY=nvapi-...`. **Never commit `.env`** (it's git-ignored).

Everyone uses **their own key**. The free limit is 40 requests/minute per key, shared by both models.

### Check everything works
```powershell
python -m pytest tests -q          # offline tests, should end with "69 passed"
python smoke_test.py               # both models + AppWorld; should end with "All good."
```
Then the task list (skip if `tasks/dev_hard.txt` is already in the repo; it's deterministic anyway):
```powershell
python select_tasks.py             # writes tasks/dev_hard.txt: 20 hard dev tasks
```

## 3. Rules that keep results valid

1. **Don't edit `score/prompts.py`, `score/agents.py` or `config.yaml` in the middle of an experiment.**
   Runs are only comparable if the code and settings are identical. Each run saves its settings
   in `runs/<run-name>/config_used.yaml`.
2. **If you change code or settings, use new run names** (e.g. add `_v4`) and re-run every
   condition you want to compare.
3. **Close the project's files in your editor while things run**, or at least don't save them.
   An editor with old copies open once silently overwrote updated files.
4. **One run at a time per API key.** Two runs on one key share the 40 requests/min and just slow each other down.
5. **Prefix run names with your name** so folders never clash when we combine results,
   e.g. `asha_score_k3_v3`.

## 4. Running

```powershell
python run.py --mode executor --run-name <you>_executor_v3 --limit 5
python run.py --mode frontier --run-name <you>_frontier_v3 --limit 5
python run.py --mode score --k 3 --run-name <you>_score_k3_v3 --limit 5
python summarize.py --runs <you>_executor_v3 <you>_frontier_v3 <you>_score_k3_v3
```

- `--limit N` = first N tasks of `tasks/dev_hard.txt`. Drop it to run all 20.
- **Re-running the same command continues where it stopped.** Finished tasks are skipped; crashed
  tasks and tasks whose grading failed are retried. So Ctrl+C is safe, but the task in progress is lost.
- Rough timings per task: executor-only and frontier-only ~1 min, ScoRe ~4–5 min.
- Other flags: `--view full|notes` (planner view), `--task-ids <id> <id>`, `--quiet`.

## 5. The experiments (split them between you)

Use the **same task list** for everything. Suggested split, one person per row:

| # | Experiment | Commands (per person, `--limit 20` or no limit) | Question it answers |
|---|---|---|---|
| E1 | Baselines | `--mode executor` and `--mode frontier` | the two reference points |
| E2 | ScoRe main | `--mode score --k 3` | does ScoRe keep success while cutting frontier tokens? |
| E3 | k sweep | `--mode score --k 2`, `--k 5`, `--k 8` (one run name each) | how does the window size trade success vs frontier cost? |
| E4 | Planner view | `--mode score --k 3 --view full` | is the notes view really better than full history? |
| E5 (optional) | Strict fixed k | set `score.allow_early_handback: false` in a copy of the config, `--config config_strict.yaml` | does letting the executor hand back early help? |

Do E1 and E2 first on 5 tasks to make sure everything works, then on all 20.

For E5, copy `config.yaml` to `config_strict.yaml`, change only that line, and pass
`--config config_strict.yaml`. Don't edit the shared `config.yaml`.

## 6. Reading the results

`python summarize.py` compares runs **on the tasks they have all finished** and writes `runs/summary.csv`.

| Column | Meaning |
|---|---|
| `success_rate` | fraction of tasks where every AppWorld check passed |
| `avg_pass_pct` | average % of checks passed (partial credit) |
| `avg_env_steps` | code executions per task |
| `avg_frontier_calls`, `avg_frontier_tokens` | frontier model usage (the thing ScoRe should cut) |
| `avg_cheap_calls`, `avg_cheap_tokens` | 20B usage |
| `avg_fetches` | how often the planner re-read an old output (notes view) |
| `avg_rejections` | steps the pre-execution checker stopped (and corrected) |
| `est_usd_per_task` | cost at listed prices (Nemotron price is a placeholder) |
| `avg_est_tflops`, `rel_tflops` | compute estimate (2 × active params × tokens) |
| `rel_cost_R2/5/10/20` | cost relative to frontier-only if the frontier model were 2/5/10/20× pricier per token |
| `trimmed_calls`, `estimated_usage_calls` | should be 0 on NVIDIA; non-zero means a fallback path was used |

Per task, `runs/<run-name>/<task_id>.json` has everything: every step's code and output, every
model call (tokens, latency, reasoning, notes such as `rejected` or `fetch:12`), and for ScoRe
each window's instructions, the planner's progress notes and the handback reason.
AppWorld's own log of every executed step is in
`experiments/outputs/score_<run-name>/tasks/<task_id>/logs/environment_io.md`.

When something looks odd, read the task JSON's `windows` list first: it shows what the planner
asked for and how each window ended.

## 7. Sharing results

`runs/` is git-ignored by default. To share:

- **Simplest:** send `runs/summary.csv` plus a zip of your `runs/<you>_*` folders.
- **Via git:** delete the `runs/` line from `.gitignore` in your branch and commit only your own
  `runs/<you>_*` folders. Because of the name prefixes, nobody's folders clash, and
  `python summarize.py` on the combined folder compares everything.

Never commit `.env`, `data/` or `experiments/` (large, and AppWorld asks that its data not be
published unencrypted).

## 8. Troubleshooting (problems we already hit)

| Symptom | Cause / fix |
|---|---|
| `HTTP 410 … reached its end of life` | NVIDIA retired that model. Run `python probe_models.py` to see live models; set one with `python probe_models.py --use <model-id>`, and update its `price_per_m` / `active_params_b` in `config.yaml`. |
| `HTTP 404` for a model that should exist / the log shows an old model name | `config.yaml` was overwritten by an editor or a stale copy. Check the `model:` lines. |
| `probe_models.py` seems stuck | very large models queue on the free tier; it gives up after `--timeout` seconds (default 90) per reply. |
| `module 'signal' has no attribute 'SIGALRM'` | old code. Pull the latest; Windows is handled now. |
| `'charmap' codec can't encode` in `eval_error` | old code. Pull the latest; scripts restart themselves in UTF-8 mode on Windows. |
| `tool_use_failed` / "recovered code from a malformed tool call" | gpt-oss sometimes calls its built-in `python` tool; the code is recovered automatically and the call flagged. Harmless. |
| `request_too_large`, `429` warnings | only on Groq's free tier (8K tokens/min). The default config uses NVIDIA, where this shouldn't happen. |
| "rejected before running: …" in the log | the pre-execution checker caught a wrong API/parameter/variable and sent the model a correction. Expected; counted in `avg_rejections`. |
| "Daily quota exhausted" | re-run the same command later; it continues. |
| `smoke_test.py` AppWorld check fails | run `appworld install` and `appworld download data` inside the repo folder. |
| `pip install` / `appworld install` fails on Windows | do the setup inside WSL (Ubuntu) with the Linux commands. |

## 9. Known limitations (say these in the write-up)

- Results so far are from **one task**. Nothing is conclusive until E1/E2 run on all 20.
- Planner and executor are from **different model families** (NVIDIA vs OpenAI), because NVIDIA retired gpt-oss-120b.
- Nemotron's **price is a placeholder**. Rely on tokens, calls and the `rel_cost_R*` columns rather than dollars.
- NVIDIA doesn't report hardware timing, so those columns are 0. `avg_latency_s` (our own measurement) is real.
- The pre-execution checker applies to **all** modes, so the baselines get it too (fair comparison).
- The current code is built around AppWorld's style of task (agents write Python against an `apis`
  object). Supporting an arbitrary task with an arbitrary tool list needs a second, generic
  tool-calling environment. The core loop is reusable; see `score/context.py` for the hook interface.

## 10. Where things are

| File | What it does |
|---|---|
| `run.py` | runs one condition over the task list |
| `summarize.py` | compares runs → `runs/summary.csv` |
| `config.yaml` | models, k, planner view, budgets |
| `probe_models.py` | lists live NVIDIA models, tests them, sets the planner |
| `smoke_test.py`, `select_tasks.py` | setup checks, task list |
| `score/agents.py` | the loops: single-model baseline and ScoRe planner/executor |
| `score/prompts.py` | all prompts |
| `score/context.py` | task-agnostic state/reference/check hooks |
| `score/env.py` | AppWorld adapter: execution, checker, API reference, variable list |
| `score/llm.py` | API client: throttling, retries, usage/timing tracking |
| `score/state.py` | global/local state and the trajectory log |
| `tests/` | offline tests (fake environment + scripted models) |
