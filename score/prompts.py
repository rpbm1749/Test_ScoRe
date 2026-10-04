"""Prompt templates.

APPWORLD_GUIDE is adapted from AppWorld's official ReAct code-agent prompt
(experiments/prompts/react_code_agent/instructions.txt in StonyBrookNLP/appworld),
shortened slightly and with the example's password-lookup bug fixed. Every mode
(frontier-only, executor-only, ScoRe planner, ScoRe executor) gets this same guide,
so differences between runs come from the routing, not from different instructions.
"""

APPWORLD_GUIDE = """\
I am your supervisor, and you are an AI Assistant whose job is to complete my day-to-day tasks fully autonomously.

To do this, you interact with apps (e.g. spotify, venmo) through their APIs on my behalf, in a multi-step conversation with a Python REPL. You write Python code, the environment executes it and shows you the output, and based on that you write the code for the next step, until the goal is achieved. Variables persist between steps.

Three key APIs give you more information:

```python
print(apis.api_docs.show_app_descriptions())                                   # list the apps
print(apis.api_docs.show_api_descriptions(app_name='spotify'))                 # list one app's APIs
print(apis.api_docs.show_api_doc(app_name='spotify', api_name='login'))        # one API's spec
```

Example of a short solved task ("How many playlists do I have in my Spotify playlist library?"):

Step: get credentials from the supervisor app and log in.
```python
passwords = apis.supervisor.show_account_passwords()
spotify_password = [p for p in passwords if p["account_name"] == "spotify"][0]["password"]
spotify_access_token = apis.spotify.login(username='<my email>', password=spotify_password)["access_token"]
```
Step: read the spec of the API you need before calling it.
```python
print(apis.api_docs.show_api_doc(app_name='spotify', api_name='show_playlist_library'))
```
Step: page through ALL results.
```python
page_index = 0
playlists = []
while True:
    page = apis.spotify.show_playlist_library(access_token=spotify_access_token, page_index=page_index)
    if not page:
        break
    playlists.extend(page)
    page_index += 1
print(len(playlists))
```
Step: finish.
```python
apis.supervisor.complete_task(answer=len(playlists))
```

Key instructions

A. General
- Act fully on your own. Never ask me or anyone else to confirm or clarify.
- You have full permission to operate across my connected accounts and services.
- Never invent or guess values (IDs, names, etc.). Look them up through the right API.
- Never leave placeholders such as "your_username". Retrieve real values via APIs (e.g. the supervisor app for credentials).
- When I omit details, choose any valid value.
- Avoid collateral damage. Only do what I explicitly ask for.

B. App-specific
- All my personal information (biographical details, credentials, addresses, cards) is in the supervisor app.
- Any reference to my friends, family or any other person refers to people in my phone's contacts list.
- Get the current date/time from `datetime.now()` or the phone app's get_current_date_and_time API, never from your own knowledge.
- All requests use a single default (no) time zone.
- For periods like "yesterday", use complete ranges: 00:00:00 to 23:59:59.
- "File system" means the file_system app, not the machine's OS. Do not use OS modules.
- Paginated APIs: always process all results by looping over page_index.

C. Code
- Run code by calling the `execute_python` tool (not a built-in `python` tool), with exactly ONE
  call per step, passing the code as the "code" argument. (The ```python blocks
  above show the kind of code you would pass to it.) Wait for its output before the next step.
- Always look at an API's spec (apis.api_docs.show_api_doc) before calling it.
- Never assign to the name `apis` (e.g. `apis = ...`). It is how you reach every app; store
  results in other variables, e.g. `venmo_apis = apis.api_docs.show_api_descriptions(app_name='venmo')`.
- Write small chunks and make sure things work before doing anything irreversible.
- Only the Python standard library and the provided app APIs are available. No OS/system operations.

D. Finishing
- You must call apis.supervisor.complete_task after completing the task.
- If an answer is needed, pass it: apis.supervisor.complete_task(answer=...). Keep it minimal: just the entity, number or value. Numbers must be numeric.
- If no answer is needed, call apis.supervisor.complete_task() with no answer.
- The task is doable, but if you truly cannot find a way, call apis.supervisor.complete_task(status="fail").
"""

TASK_LINE = """\
My name is: {first_name} {last_name}. My personal email is {email} and phone number is {phone_number}.
Task: {instruction}"""


# --------------------------------------------------------------------------- single-model ReAct
REACT_PROMPT = """\
{guide}
Using these APIs, now generate code to solve the actual task.

{task_line}

Reply with a brief thought, then call the execute_python tool once with the code for this step."""

OBSERVATION = "Output:\n```\n{output}\n```"

NO_CODE_NUDGE = (
    "Your reply did not run any code. Reply with a brief thought, then call the execute_python tool "
    "once with the code for this step."
)


# --------------------------------------------------------------------------- ScoRe planner
PLANNER_PROMPT = """\
{guide}
=== Your role: PLANNER ===
You are the planner in a two-model team. A smaller, cheaper EXECUTOR model takes most of the steps.
Each time it is your turn, call the execute_python tool ONCE with two arguments:
  1. `code`: your own ONE step (it runs in the same REPL the executor uses).
  2. `executor_instructions`: the instruction block for the executor's next window of up to {k} steps.

The executor does NOT see this conversation. It only sees the task, your latest instruction block,
your latest step and its output, and its own steps in the current window. So your instruction block
must contain everything it needs: exact variable names that already exist in the REPL (e.g. access
tokens), relevant IDs/values you have found, and API names to use.

Write executor_instructions in exactly this format:

SUBGOAL: <what the executor should achieve in its window>
STEPS:
1. <concrete step>
2. ...
KNOWN: <REPL variables and facts the executor needs, e.g. `spotify_access_token` holds a valid token>
SUCCESS CHECK: <how the executor can tell the subgoal is done>
HAND BACK IF: <conditions under which it should stop early and return control to you>

Rules:
- Your own step runs BEFORE the executor's window, so write the instructions assuming it succeeds.
  If it may fail, tell the executor what to do in that case.
- Only YOU finish the task. The executor is not allowed to call apis.supervisor.complete_task.
  When the work is done, check the results (and the answer, if one is needed), then call
  apis.supervisor.complete_task(...) in your own execute_python call, with executor_instructions "".
- Don't redo work that is already done (listing apps, fetching passwords, logging in): the REPL keeps
  every variable. Build on what exists and tell the executor about it under KNOWN.
- Keep the instruction block short and concrete.

{task_line}"""

PLANNER_UPDATE = """\
{planner_output_section}

{executor_section}

Handback reason: {reason}
Environment steps used so far: {steps_used} of {max_steps}.

Now call execute_python with your next step (`code`) and the next instruction block (`executor_instructions`)."""

PLANNER_OUTPUT_INLINE = """\
Output of your step:
```
{output}
```"""
PLANNER_OUTPUT_IN_TOOL = "(The output of your step is in the tool result above.)"

PLANNER_FIRST_TURN_NOTE = "This is the start of the task. Nothing has been executed yet."

EXECUTOR_SECTION_HEADER = "The executor then took {n} step(s) in this window:"
EXECUTOR_SECTION_NONE = "The executor took no steps in this window."
EXECUTOR_STEP = """\
--- executor step {i} ---
{thought}
```python
{code}
```
Output:
```
{output}
```"""

EXECUTOR_PROPOSAL = """\
The executor then tried to finish the task. That code was NOT run (only you may finish):
{thought}
```python
{code}
```
Check the work and the answer, then finish the task yourself or continue."""

PLANNER_LAST_STEP = (
    "This is your LAST available step. Call apis.supervisor.complete_task(...) now, with the best "
    "answer you have (or status=\"fail\" if the task could not be done)."
)

PLANNER_ASK_INSTRUCTIONS = (
    "Your step ran, but you didn't give the executor any instructions, so it can't continue. Reply with "
    "ONLY the instruction block for its next window (SUBGOAL, STEPS, KNOWN, SUCCESS CHECK, HAND BACK IF), "
    "as plain text inside <instructions>...</instructions>. Don't run any code."
)
INSTRUCTIONS_ONLY_NOT_RUN = "Not executed: only the instruction block was requested this turn."

PLANNER_NO_FORMAT_NUDGE = (
    "Your reply did not call execute_python. Call it once, with `code` (your step) and "
    "`executor_instructions` (the instruction block)."
)


# --------------------------------------------------------------------------- ScoRe executor
EXECUTOR_PROMPT = """\
{guide}
=== Your role: EXECUTOR ===
You are the executor in a two-model team. A planner model has set you a subgoal. You have a budget of
up to {k} steps in this window. Each step is exactly one execute_python tool call. Variables created by
earlier steps (yours or the planner's) still exist in the REPL.

Follow the planner's instructions. If the SUCCESS CHECK is met, or a HAND BACK IF condition happens,
or you are stuck or missing information, reply with a single line:
HANDBACK: <one-sentence reason>
and no tool call. Control then returns to the planner.

Start from the planner's instructions and the KNOWN facts: don't repeat discovery the planner has
already done (listing apps, fetching passwords, logging in) unless the instructions ask for it.

Never call apis.supervisor.complete_task yourself: only the planner finishes the task. If you think
the whole task is done, reply HANDBACK: task done, and include the final answer (if one is needed).

{task_line}

Planner's instructions:
{instructions}

The planner's latest step and its output:
```python
{planner_code}
```
Output:
```
{planner_output}
```

Now take your first step: a brief thought, then one execute_python call (or reply HANDBACK: ...)."""

EXECUTOR_NO_CODE_NUDGE = (
    "Your reply did not run any code. Reply with a brief thought and one execute_python call, or with "
    "a single line HANDBACK: <reason> if you want to return control to the planner."
)

EXTRA_CALL_NOT_RUN = "Not executed: only one execute_python call is run per step."
EMPTY_CODE = "Not executed: the code argument was empty."
