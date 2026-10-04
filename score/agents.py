"""The three ways of running a task.

run_single   - ReAct with one model. Used for both baselines:
               frontier-only (120B every step) and executor-only (20B every step).
run_score    - ScoRe: the planner takes one step and writes an instruction block,
               then the executor takes up to k steps, then control returns to the planner.

All three return (end_reason, Trajectory, extra_info). Evaluation happens in run.py.
"""

from __future__ import annotations

from typing import Any

from . import prompts as P
from .llm import LLMClient
from .parsing import (
    extract_code,
    extract_handback,
    extract_instructions,
    is_error_output,
    strip_instructions,
    thought_before_code,
    truncate,
)
from .state import GlobalState, LocalState, StepRecord, Trajectory, WindowRecord


def _task_line(env: Any) -> str:
    sup = env.supervisor
    return P.TASK_LINE.format(
        first_name=sup.get("first_name", ""),
        last_name=sup.get("last_name", ""),
        email=sup.get("email", ""),
        phone_number=sup.get("phone_number", ""),
        instruction=env.instruction,
    )


def _attach(e: BaseException, traj: Trajectory) -> BaseException:
    if not hasattr(e, "traj"):
        e.traj = traj  # type: ignore[attr-defined]
    return e


def _log(verbose: bool, msg: str) -> None:
    if verbose:
        print(msg, flush=True)


def _short(code: str, n: int = 90) -> str:
    first = code.strip().splitlines()[0] if code.strip() else ""
    return first[:n] + ("..." if len(first) > n or len(code.strip().splitlines()) > 1 else "")


# =============================================================================== single model
def run_single(env: Any, llm: LLMClient, cfg: dict[str, Any], role: str, verbose: bool = True) -> tuple[str, Trajectory, dict]:
    run_cfg = cfg["run"]
    max_chars = run_cfg["max_output_chars"]
    max_no_code = run_cfg["max_no_code_retries"]

    traj = Trajectory(env.task_id, mode=f"{role}_only", k=None)
    messages = [{"role": "user", "content": P.REACT_PROMPT.format(guide=P.APPWORLD_GUIDE, task_line=_task_line(env))}]
    try:
        return _single_loop(env, llm, role, messages, traj, max_chars, max_no_code, verbose)
    except Exception as e:
        raise _attach(e, traj)


def _single_loop(env, llm, role, messages, traj, max_chars, max_no_code, verbose):
    no_code = 0
    while True:
        if env.completed():
            return "completed", traj, {}
        if env.out_of_steps:
            return "max_steps", traj, {}
        res = llm.chat(messages)
        code = extract_code(res.content)
        traj.add_call(role, res.usage, None, produced_code=bool(code))
        messages.append({"role": "assistant", "content": res.content})
        if not code:
            no_code += 1
            _log(verbose, f"  [{role}] no code block ({no_code}/{max_no_code})")
            if no_code >= max_no_code:
                return "no_code", traj, {}
            messages.append({"role": "user", "content": P.NO_CODE_NUDGE})
            continue
        no_code = 0
        output = env.execute(code)
        shown = truncate(output, max_chars)
        traj.add_step(StepRecord(env.steps, role, None, thought_before_code(res.content), code, output, shown, is_error_output(output)))
        _log(verbose, f"  step {env.steps:>2} [{role}] {_short(code)}{'  -> ERROR' if is_error_output(output) else ''}")
        messages.append({"role": "user", "content": P.OBSERVATION.format(output=shown)})


# =============================================================================== ScoRe
def _planner_turn(
    env: Any, planner: LLMClient, gstate: GlobalState, traj: Trajectory, window: int, cfg: dict[str, Any], verbose: bool
) -> tuple[str | None, str, str, str | None]:
    """Ask the planner for an instruction block + one step. Returns (instructions, code, output, end_reason)."""
    max_no_code = cfg["run"]["max_no_code_retries"]
    max_chars = cfg["run"]["max_output_chars"]
    failures = 0
    while True:
        res = planner.chat(gstate.planner_messages)
        instructions = extract_instructions(res.content)
        code = extract_code(strip_instructions(res.content))
        traj.add_call("planner", res.usage, window, produced_code=bool(code), note="" if instructions else "no_instructions")
        gstate.planner_messages.append({"role": "assistant", "content": res.content})
        if instructions or code:
            break
        failures += 1
        _log(verbose, f"  [planner] reply had no instructions or code ({failures}/{max_no_code})")
        if failures >= max_no_code:
            return None, "", "", "no_code"
        gstate.planner_messages.append({"role": "user", "content": P.PLANNER_NO_FORMAT_NUDGE})

    output = ""
    if code:
        if env.out_of_steps:
            return instructions, code, "", "max_steps"
        output = env.execute(code)
        shown = truncate(output, max_chars)
        thought = thought_before_code(strip_instructions(res.content))
        traj.add_step(StepRecord(env.steps, "planner", window, thought, code, output, shown, is_error_output(output)))
        _log(verbose, f"  step {env.steps:>2} [planner  w{window}] {_short(code)}{'  -> ERROR' if is_error_output(output) else ''}")
        output = shown
    else:
        output = "(the planner took no step this turn)"
    return instructions, code, output, None


def _executor_window(
    env: Any, executor: LLMClient, gstate: GlobalState, lstate: LocalState, traj: Trajectory, k: int, cfg: dict[str, Any], verbose: bool
) -> str:
    """Run up to k executor steps. Returns the handback reason."""
    score_cfg = cfg["score"]
    max_chars = cfg["run"]["max_output_chars"]
    max_no_code = cfg["run"]["max_no_code_retries"]
    err_limit = score_cfg.get("handback_on_consecutive_errors") or 0
    allow_early = score_cfg.get("allow_early_handback", True)

    lstate.messages = [{
        "role": "user",
        "content": P.EXECUTOR_PROMPT.format(
            guide=P.APPWORLD_GUIDE,
            k=k,
            task_line=gstate.task_line,
            instructions=lstate.instructions,
            planner_code=lstate.planner_code or "# (no step)",
            planner_output=lstate.planner_output,
        ),
    }]
    no_code = 0
    while lstate.steps_taken < k:
        if env.completed():
            return "task_completed"
        if env.out_of_steps:
            return "max_steps"
        res = executor.chat(lstate.messages)
        code = extract_code(res.content)
        handback = extract_handback(res.content) if allow_early else None
        traj.add_call("executor", res.usage, lstate.window, produced_code=bool(code), note="handback" if handback else "")
        lstate.messages.append({"role": "assistant", "content": res.content})
        if handback:
            _log(verbose, f"  [executor w{lstate.window}] HANDBACK: {handback}")
            return f"executor_handback: {handback}"
        if not code:
            no_code += 1
            if no_code >= max_no_code:
                return "executor_no_code"
            lstate.messages.append({"role": "user", "content": P.EXECUTOR_NO_CODE_NUDGE})
            continue
        no_code = 0
        output = env.execute(code)
        shown = truncate(output, max_chars)
        err = is_error_output(output)
        rec = StepRecord(env.steps, "executor", lstate.window, thought_before_code(res.content), code, output, shown, err)
        traj.add_step(rec)
        lstate.step_records.append(rec)
        lstate.steps_taken += 1
        _log(verbose, f"  step {env.steps:>2} [executor w{lstate.window}] {_short(code)}{'  -> ERROR' if err else ''}")
        lstate.consecutive_errors = lstate.consecutive_errors + 1 if err else 0
        if err_limit and lstate.consecutive_errors >= err_limit:
            return f"consecutive_errors ({err_limit})"
        lstate.messages.append({"role": "user", "content": P.OBSERVATION.format(output=shown)})
    if env.completed():
        return "task_completed"
    return f"budget_exhausted (k={k})"


def _planner_update_message(lstate: LocalState | None, planner_output: str, reason: str, env: Any) -> str:
    if lstate is None or not lstate.step_records:
        section = P.EXECUTOR_SECTION_NONE
    else:
        parts = [P.EXECUTOR_SECTION_HEADER.format(n=len(lstate.step_records))]
        for i, s in enumerate(lstate.step_records, 1):
            parts.append(P.EXECUTOR_STEP.format(i=i, thought=s.thought or "(no thought)", code=s.code, output=s.output_shown))
        section = "\n\n".join(parts)
    return P.PLANNER_UPDATE.format(
        planner_output=planner_output,
        executor_section=section,
        reason=reason,
        steps_used=env.steps,
        max_steps=env.max_steps,
    )


def run_score(env: Any, planner: LLMClient, executor: LLMClient, cfg: dict[str, Any], verbose: bool = True) -> tuple[str, Trajectory, dict]:
    k = int(cfg["score"]["k"])
    traj = Trajectory(env.task_id, mode="score", k=k)
    gstate = GlobalState(task_line=_task_line(env))
    gstate.planner_messages = [{
        "role": "user",
        "content": P.PLANNER_PROMPT.format(guide=P.APPWORLD_GUIDE, k=k, task_line=gstate.task_line)
        + "\n\n" + P.PLANNER_FIRST_TURN_NOTE,
    }]

    try:
        end_reason = _score_loop(env, planner, executor, cfg, verbose, k, traj, gstate)
    except Exception as e:
        raise _attach(e, traj)
    return end_reason, traj, _windows_info(gstate)


def _windows_info(gstate: GlobalState) -> dict:
    return {
        "windows": [
            {
                "window": w.window,
                "planner_step": w.planner_step,
                "executor_steps": w.executor_steps,
                "handback_reason": w.handback_reason,
                "instructions": w.instructions,
            }
            for w in gstate.windows
        ]
    }


def _score_loop(env, planner, executor, cfg, verbose, k, traj, gstate) -> str:
    window = 0
    last_instructions = "(none yet)"
    end_reason = ""
    while True:
        if env.completed():
            end_reason = "completed"
            break
        if env.out_of_steps:
            end_reason = "max_steps"
            break
        window += 1
        instructions, code, p_output, stop = _planner_turn(env, planner, gstate, traj, window, cfg, verbose)
        wrec = WindowRecord(window=window, instructions=instructions, planner_step=env.steps if code else None)
        gstate.windows.append(wrec)
        if stop:
            end_reason = stop
            break
        if instructions:
            last_instructions = instructions
        if env.completed():
            end_reason = "completed"
            break

        lstate = LocalState(window=window, instructions=last_instructions, planner_code=code, planner_output=p_output)
        if k > 0:
            reason = _executor_window(env, executor, gstate, lstate, traj, k, cfg, verbose)
        else:
            reason = "k=0 (no executor)"
        wrec.executor_steps = [s.step for s in lstate.step_records]
        wrec.handback_reason = reason
        gstate.planner_messages.append({"role": "user", "content": _planner_update_message(lstate, p_output, reason, env)})

    return end_reason
