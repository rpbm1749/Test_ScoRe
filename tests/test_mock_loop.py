"""Offline tests of the orchestration loop: fake environment + scripted models.

No API key or AppWorld data needed:   python -m pytest tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from score.agents import run_score, run_single  # noqa: E402
from score.llm import ChatResult, Usage  # noqa: E402

CFG = {
    "score": {"k": 2, "allow_early_handback": True, "handback_on_consecutive_errors": 2},
    "run": {"max_steps": 20, "max_output_chars": 50, "max_no_code_retries": 2},
}


class FakeEnv:
    def __init__(self, max_steps: int = 20) -> None:
        self.task_id = "fake_1"
        self.instruction = "Count my playlists."
        self.supervisor = {"first_name": "Ada", "last_name": "L", "email": "a@x.com", "phone_number": "1"}
        self.max_steps = max_steps
        self.steps = 0
        self.done = False
        self.codes: list[str] = []

    def execute(self, code: str) -> str:
        self.steps += 1
        self.codes.append(code)
        if "complete_task" in code:
            self.done = True
            return "Marked the active task complete."
        if "boom" in code:
            return "Execution failed. Traceback:\nNameError: boom"
        return f"output of step {self.steps} " + "x" * 100  # long, to exercise truncation

    def completed(self) -> bool:
        return self.done

    @property
    def out_of_steps(self) -> bool:
        return self.steps >= self.max_steps

    def evaluate(self) -> dict:
        return {"success": self.done, "pass_percentage": 100.0 if self.done else 0.0}

    def close(self) -> None:
        pass


class ToolReply:
    """A scripted reply that calls execute_python (optionally several times)."""

    def __init__(self, *codes: str, text: str = "thinking", instructions: str | None = None, reasoning: str = "",
                 notes: str | None = None, fetch: int | None = None) -> None:
        self.codes = codes
        self.text = text
        self.instructions = instructions
        self.reasoning = reasoning
        self.notes = notes
        self.fetch = fetch


_ids = iter(range(10**6))


def assert_valid_tool_sequence(messages: list[dict]) -> None:
    """API rule: every tool call is answered by exactly one tool message, right after its assistant message."""
    i = 0
    while i < len(messages):
        m = messages[i]
        if m["role"] == "tool":
            raise AssertionError(f"orphan tool message at {i}: {m}")
        if m["role"] == "assistant" and m.get("tool_calls"):
            ids = [c["id"] for c in m["tool_calls"]]
            answered = []
            j = i + 1
            while j < len(messages) and messages[j]["role"] == "tool":
                answered.append(messages[j]["tool_call_id"])
                j += 1
            assert j < len(messages) or answered, "a request may not end on an unanswered tool call"
            assert sorted(answered) == sorted(ids), f"tool calls {ids} answered by {answered}"
            i = j
            continue
        i += 1


class ScriptedLLM:
    """Returns replies from a list; records every message list it was sent."""

    def __init__(self, name: str, replies: list) -> None:
        self.name = name
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def chat(self, messages, tools=None):
        assert tools and tools[0]["function"]["name"] == "execute_python"
        assert_valid_tool_sequence(messages)
        self.seen.append([dict(m) for m in messages])
        if not self.replies:
            raise AssertionError(f"{self.name} ran out of scripted replies")
        r = self.replies.pop(0)
        usage = Usage(prompt_tokens=100, completion_tokens=10, calls=1, latency_s=0.1, prompt_time_s=0.01, completion_time_s=0.02)
        if isinstance(r, ToolReply) and r.fetch is not None:
            calls = [{"id": f"call_{next(_ids)}", "type": "function",
                      "function": {"name": "get_step_output", "arguments": json.dumps({"step": r.fetch})}}]
            return ChatResult(content="", reasoning="", usage=usage, tool_calls=calls)
        if isinstance(r, ToolReply):
            calls = [
                {"id": f"call_{next(_ids)}", "type": "function",
                 "function": {"name": "execute_python", "arguments": json.dumps(
                     {"code": c, **({"executor_instructions": r.instructions} if r.instructions is not None else {}),
                      **({"progress_notes": r.notes} if r.notes is not None else {})})}}
                for c in r.codes
            ]
            return ChatResult(content=r.text, reasoning=r.reasoning, usage=usage, tool_calls=calls)
        return ChatResult(content=r, reasoning="", usage=usage)


def code(c: str, thought: str = "thinking") -> str:
    return f"{thought}\n```python\n{c}\n```"


def plan(c: str, subgoal: str = "do things") -> str:
    return f"<instructions>\nSUBGOAL: {subgoal}\nSTEPS:\n1. a\nKNOWN: tok\nSUCCESS CHECK: x\nHAND BACK IF: y\n</instructions>\n\n" + code(c)


# --------------------------------------------------------------------------- single model
def test_single_model_completes():
    env = FakeEnv()
    llm = ScriptedLLM("agent", [code("a=1"), "no code here", code("b=2"), code("apis.supervisor.complete_task(answer=3)")])
    reason, traj, _ = run_single(env, llm, CFG, role="frontier", verbose=False)
    assert reason == "completed"
    assert env.steps == 3
    assert traj.usage["frontier"].calls == 4          # includes the reply with no code
    assert traj.steps_by_actor() == {"frontier": 3}
    # outputs shown to the model are truncated, raw ones are kept
    assert "truncated" in traj.steps[0].output_shown
    assert "truncated" not in traj.steps[0].output


def test_single_model_gives_up_without_code():
    env = FakeEnv()
    llm = ScriptedLLM("agent", ["hmm", "still no code"])
    reason, _, _ = run_single(env, llm, CFG, role="executor", verbose=False)
    assert reason == "no_code"


def test_single_model_max_steps():
    env = FakeEnv(max_steps=3)
    llm = ScriptedLLM("agent", [code(f"x={i}") for i in range(5)])
    reason, traj, _ = run_single(env, llm, CFG, role="frontier", verbose=False)
    assert reason == "max_steps" and env.steps == 3


# --------------------------------------------------------------------------- ScoRe
def test_score_full_cycle():
    env = FakeEnv()
    planner = ScriptedLLM("planner", [
        plan("tok = login()", "log in and list"),          # window 1: planner step 1
        plan("y = 2", "finish listing"),                    # window 2: planner step 4
        code("apis.supervisor.complete_task(answer=5)"),    # window 3: planner finishes (no instructions needed)
    ])
    executor = ScriptedLLM("executor", [
        code("page0 = list()"),                             # step 2
        code("page1 = list()"),                             # step 3 -> budget k=2 exhausted
        "HANDBACK: subgoal done, count is 5",               # window 2 early handback
    ])
    reason, traj, extra = run_score(env, planner, executor, CFG, verbose=False)
    assert reason == "completed"
    assert [s.actor for s in traj.steps] == ["planner", "executor", "executor", "planner", "planner"]
    assert traj.usage["planner"].calls == 3 and traj.usage["executor"].calls == 3
    w = extra["windows"]
    assert w[0]["handback_reason"].startswith("budget_exhausted")
    assert w[1]["handback_reason"].startswith("executor_handback")
    assert w[0]["executor_steps"] == [2, 3]

    # Global state: the planner's 2nd call contains the executor's steps from window 1
    second_call = planner.seen[1]
    update = second_call[-1]["content"]
    assert "page0 = list()" in update and "page1 = list()" in update
    assert "Handback reason: budget_exhausted" in update

    # Local state: the executor's window-2 context has NO window-1 executor steps,
    # but does have the planner's latest instructions and step.
    exec_w2_first_call = executor.seen[2]
    joined = "\n".join(m["content"] for m in exec_w2_first_call)
    assert "page0" not in joined and "page1" not in joined
    assert "finish listing" in joined and "y = 2" in joined
    # ...and within a window the executor sees its own previous steps
    exec_w1_second_call = executor.seen[1]
    assert any("page0 = list()" in m["content"] for m in exec_w1_second_call)


def test_score_consecutive_errors_hand_back():
    env = FakeEnv()
    planner = ScriptedLLM("planner", [plan("a=1"), code("apis.supervisor.complete_task()")])
    executor = ScriptedLLM("executor", [code("boom()"), code("boom()")])
    cfg = {**CFG, "score": {**CFG["score"], "k": 5}}
    reason, traj, extra = run_score(env, planner, executor, cfg, verbose=False)
    assert reason == "completed"
    assert extra["windows"][0]["handback_reason"].startswith("consecutive_errors")
    assert len(executor.seen) == 2


def test_executor_cannot_finish_planner_does():
    env = FakeEnv()
    planner = ScriptedLLM("planner", [plan("a=1"), code("apis.supervisor.complete_task(answer=1)")])
    executor = ScriptedLLM("executor", [code("b=2"), code("apis.supervisor.complete_task(answer=1)", "all done")])
    reason, traj, extra = run_score(env, planner, executor, CFG, verbose=False)
    assert reason == "completed"
    # the executor's complete_task was never executed; the planner ran the last step
    assert not any("complete_task" in c for c in env.codes[:-1])
    assert traj.steps[-1].actor == "planner" and "complete_task" in traj.steps[-1].code
    assert extra["windows"][0]["handback_reason"] == "executor_ready_to_finish"
    update = planner.seen[1][-1]["content"]
    assert "was NOT run" in update and "complete_task(answer=1)" in update


def test_last_step_reserved_for_planner():
    env = FakeEnv(max_steps=4)
    planner = ScriptedLLM("planner", [plan("p=1"), code("apis.supervisor.complete_task(answer=2)")])
    executor = ScriptedLLM("executor", [code(f"e={i}") for i in range(10)])
    cfg = {**CFG, "score": {**CFG["score"], "k": 10}}
    reason, traj, extra = run_score(env, planner, executor, cfg, verbose=False)
    assert reason == "completed" and env.steps == 4
    assert [s.actor for s in traj.steps] == ["planner", "executor", "executor", "planner"]
    assert extra["windows"][0]["handback_reason"] == "step_budget_reserved_for_planner"
    assert "LAST available step" in planner.seen[1][-1]["content"]


def test_score_k0_is_planner_only():
    env = FakeEnv()
    planner = ScriptedLLM("planner", [plan("a=1"), plan("apis.supervisor.complete_task()")])
    executor = ScriptedLLM("executor", [])
    cfg = {**CFG, "score": {**CFG["score"], "k": 0}}
    reason, traj, _ = run_score(env, planner, executor, cfg, verbose=False)
    assert reason == "completed" and "executor" not in traj.usage


def test_score_planner_bad_format_gives_up():
    env = FakeEnv()
    planner = ScriptedLLM("planner", ["I am thinking", "still thinking"])
    reason, _, _ = run_score(env, planner, ScriptedLLM("executor", []), CFG, verbose=False)
    assert reason == "no_code"


def test_score_respects_max_steps():
    env = FakeEnv(max_steps=4)
    planner = ScriptedLLM("planner", [plan(f"p={i}") for i in range(5)])
    executor = ScriptedLLM("executor", [code(f"e={i}") for i in range(10)])
    reason, traj, _ = run_score(env, planner, executor, CFG, verbose=False)
    assert reason == "max_steps" and env.steps == 4
    assert traj.steps[-1].actor == "planner"


# --------------------------------------------------------------------------- run.py end to end
def test_run_py_end_to_end(tmp_path, monkeypatch):
    import run as runmod

    planner = ScriptedLLM("planner", [plan("a=1"), code("apis.supervisor.complete_task(answer=1)")] * 2)
    executor = ScriptedLLM("executor", [code("b=1"), "HANDBACK: done"] * 2)
    monkeypatch.setattr(runmod, "ROOT", tmp_path)
    monkeypatch.setattr(runmod, "make_env", lambda task_id, run_name, max_steps: FakeEnv(max_steps))
    monkeypatch.setattr(runmod, "build_client", lambda role, *a, **k: planner if role in ("planner", "frontier") else executor)
    (tmp_path / "config.yaml").write_text((ROOT / "config.yaml").read_text())
    monkeypatch.setattr(sys, "argv", ["run.py", "--mode", "score", "--k", "2", "--run-name", "t", "--task-ids", "t1", "t2", "--quiet"])
    runmod.main()
    rows = [json.loads(l) for l in (tmp_path / "runs/t/results.jsonl").read_text().splitlines()]
    assert [r["task_id"] for r in rows] == ["t1", "t2"]
    assert all(r["success"] for r in rows)
    assert rows[0]["frontier_calls"] == 2 and rows[0]["cheap_calls"] == 2 and rows[0]["k"] == 2
    traj = json.loads((tmp_path / "runs/t/t1.json").read_text())
    assert traj["windows"][0]["handback_reason"].startswith("executor_handback")

    # resume: a second invocation skips finished tasks
    runmod.main()
    assert len((tmp_path / "runs/t/results.jsonl").read_text().splitlines()) == 2

    # summarize.py works on the output
    import summarize

    monkeypatch.setattr(summarize, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["summarize.py"])
    summarize.main()
    assert (tmp_path / "runs/summary.csv").exists()


# --------------------------------------------------------------------------- tool calls
def tplan(c: str, subgoal: str = "do things") -> ToolReply:
    return ToolReply(c, text=f"<instructions>\nSUBGOAL: {subgoal}\nSTEPS:\n1. a\nKNOWN: tok\nSUCCESS CHECK: x\nHAND BACK IF: y\n</instructions>\nmy step")


def test_single_model_with_tool_calls():
    env = FakeEnv()
    llm = ScriptedLLM("agent", [ToolReply("a=1"), ToolReply("x=1", "y=2"), "no code", ToolReply("apis.supervisor.complete_task()")])
    reason, traj, _ = run_single(env, llm, CFG, role="executor", verbose=False)
    assert reason == "completed"
    assert env.codes == ["a=1", "x=1", "apis.supervisor.complete_task()"]   # only the first call of a reply runs
    msgs = llm.seen[-1]
    tool_msgs = [m for m in msgs if m["role"] == "tool"]
    assert "output of step 1" in tool_msgs[0]["content"]
    assert any("only one execute_python call" in m["content"] for m in tool_msgs)


def test_score_with_tool_calls():
    env = FakeEnv()
    planner = ScriptedLLM("planner", [tplan("tok = login()"), ToolReply("apis.supervisor.complete_task(answer=2)")])
    executor = ScriptedLLM("executor", [ToolReply("p = 1"), ToolReply("apis.supervisor.complete_task(answer=2)")])
    reason, traj, extra = run_score(env, planner, executor, CFG, verbose=False)
    assert reason == "completed"
    assert [s.actor for s in traj.steps] == ["planner", "executor", "planner"]
    assert extra["windows"][0]["handback_reason"] == "executor_ready_to_finish"
    assert extra["windows"][0]["instructions"].startswith("SUBGOAL")
    second = planner.seen[1]
    assert second[2]["role"] == "tool" and "output of step 1" in second[2]["content"]
    assert "tool result above" in second[-1]["content"] and "p = 1" in second[-1]["content"]
    # the executor got the planner's step and output in its prompt
    assert "tok = login()" in executor.seen[0][0]["content"]


def test_tool_call_with_empty_code_is_nudged():
    env = FakeEnv()
    llm = ScriptedLLM("agent", [ToolReply(""), ToolReply("apis.supervisor.complete_task()")])
    reason, traj, _ = run_single(env, llm, CFG, role="frontier", verbose=False)
    assert reason == "completed" and env.steps == 1


def test_planner_tool_call_with_empty_code_then_executor():
    env = FakeEnv()
    planner = ScriptedLLM("planner", [tplan(""), ToolReply("apis.supervisor.complete_task()")])
    executor = ScriptedLLM("executor", [ToolReply("e=1"), "HANDBACK: done"])
    reason, traj, _ = run_score(env, planner, executor, CFG, verbose=False)
    assert reason == "completed"
    assert [s.actor for s in traj.steps] == ["executor", "planner"]


def test_errored_tasks_are_retried(tmp_path):
    import run as runmod

    path = tmp_path / "results.jsonl"
    path.write_text(
        json.dumps({"task_id": "a", "end_reason": "error"}) + "\n"
        + json.dumps({"task_id": "b", "end_reason": "completed"}) + "\n"
        + json.dumps({"task_id": "c", "end_reason": "request_too_large"}) + "\n"
        + json.dumps({"task_id": "d", "end_reason": "completed", "eval_error": "charmap"}) + "\n"
    )
    assert runmod.done_task_ids(path) == {"b", "c"}
    with open(path, "a") as f:
        f.write(json.dumps({"task_id": "a", "end_reason": "completed"}) + "\n")
    assert runmod.done_task_ids(path) == {"a", "b", "c"}


# --------------------------------------------------------------------------- planner instructions
INSTR = "SUBGOAL: find Kristin\nSTEPS:\n1. search contacts\nKNOWN: `phone_token` is valid\nSUCCESS CHECK: id found\nHAND BACK IF: none"


def test_planner_instructions_from_tool_argument():
    env = FakeEnv()
    planner = ScriptedLLM("planner", [ToolReply("phone_token = 1", text="", instructions=INSTR),
                                      ToolReply("apis.supervisor.complete_task()", text="", instructions="")])
    executor = ScriptedLLM("executor", ["HANDBACK: done"])
    reason, traj, extra = run_score(env, planner, executor, CFG, verbose=False)
    assert reason == "completed"
    assert extra["windows"][0]["instructions"] == INSTR
    assert "find Kristin" in executor.seen[0][0]["content"]
    assert len(planner.seen) == 2  # no follow-up needed


def test_planner_without_instructions_is_asked_once():
    env = FakeEnv()
    planner = ScriptedLLM("planner", [
        ToolReply("phone_token = 1", text=""),                         # step, but no instructions anywhere
        f"<instructions>\n{INSTR}\n</instructions>",                  # answer to the follow-up
        ToolReply("apis.supervisor.complete_task()", text="", instructions=""),
    ])
    executor = ScriptedLLM("executor", ["HANDBACK: done"])
    reason, traj, extra = run_score(env, planner, executor, CFG, verbose=False)
    assert reason == "completed"
    assert extra["windows"][0]["instructions"] == INSTR
    assert "find Kristin" in executor.seen[0][0]["content"]
    notes = [c.note for c in traj.calls if c.role == "planner"]
    assert "instructions_followup" in notes
    assert "didn't give the executor any instructions" in planner.seen[1][-1]["content"]
    assert env.codes.count("phone_token = 1") == 1  # the follow-up didn't re-run anything


def test_followup_tool_call_is_answered_not_run():
    env = FakeEnv()
    planner = ScriptedLLM("planner", [
        ToolReply("a = 1", text=""),
        ToolReply("a = 2", text="", instructions=INSTR),   # follow-up answered with another tool call
        ToolReply("apis.supervisor.complete_task()", text="", instructions=""),
    ])
    executor = ScriptedLLM("executor", ["HANDBACK: done"])
    reason, _, extra = run_score(env, planner, executor, CFG, verbose=False)
    assert reason == "completed" and "a = 2" not in env.codes
    assert extra["windows"][0]["instructions"] == INSTR


def test_planner_instructions_from_reasoning():
    env = FakeEnv()
    planner = ScriptedLLM("planner", [
        ToolReply("a = 1", text="", reasoning=f"plan...\n<instructions>\n{INSTR}\n</instructions>"),
        ToolReply("apis.supervisor.complete_task()", text="", instructions=""),
    ])
    executor = ScriptedLLM("executor", ["HANDBACK: done"])
    _, _, extra = run_score(env, planner, executor, CFG, verbose=False)
    assert extra["windows"][0]["instructions"] == INSTR


# --------------------------------------------------------------------------- notes view
NCFG = {**CFG, "score": {**CFG["score"], "k": 2, "planner_view": "notes"}}


def test_notes_view_planner_sees_only_latest_window():
    env = FakeEnv()
    planner = ScriptedLLM("planner", [
        ToolReply("p1 = 1", text="", instructions=INSTR, notes="NOTES-A: nothing found yet"),
        ToolReply("p2 = 1", text="", instructions=INSTR, notes="NOTES-B: Kristin id 42"),
        ToolReply("p3 = 1", text="", instructions=INSTR, notes="NOTES-C: paid"),
        ToolReply("apis.supervisor.complete_task()", text="", instructions="", notes="done"),
    ])
    executor = ScriptedLLM("executor", [ToolReply("e_w1_a = 1"), ToolReply("e_w1_b = 1"),
                                        ToolReply("e_w2_a = 1"), ToolReply("e_w2_b = 1"),
                                        ToolReply("e_w3_a = 1"), ToolReply("e_w3_b = 1")])
    reason, traj, extra = run_score(env, planner, executor, NCFG, verbose=False)
    assert reason == "completed"
    # every planner call is a single fresh message
    assert all(len(seen) == 1 for seen in planner.seen)
    third = planner.seen[2][0]["content"]
    assert "NOTES-B: Kristin id 42" in third and "NOTES-A" not in third       # latest notes only
    assert "e_w2_a = 1" in third and "output of step" in third                # latest window, with outputs
    assert "e_w1_a = 1" in third                                              # ...older steps only in the one-line log
    w1_output_marker = "output of step 2 "                                    # step 2 = first executor step of window 1
    assert w1_output_marker not in third
    assert "  2  executor  ok" in third                                       # step log line
    # notes recorded per window
    assert [w["notes"] for w in extra["windows"]][:3] == ["NOTES-A: nothing found yet", "NOTES-B: Kristin id 42", "NOTES-C: paid"]
    # prompt size stays bounded: the 4th call isn't much bigger than the 2nd
    sizes = [len(seen[0]["content"]) for seen in planner.seen]
    assert sizes[3] < sizes[1] * 1.3


def test_notes_view_fetch_tool():
    env = FakeEnv()
    planner = ScriptedLLM("planner", [
        ToolReply("p1 = 1", text="", instructions=INSTR, notes="n1"),
        ToolReply(fetch=2),                                   # re-read step 2's full output
        ToolReply(fetch=99),                                  # a step that doesn't exist
        ToolReply("apis.supervisor.complete_task()", text="", instructions="", notes="done"),
    ])
    executor = ScriptedLLM("executor", [ToolReply("e = 1"), "HANDBACK: done"])
    reason, traj, _ = run_score(env, planner, executor, NCFG, verbose=False)
    assert reason == "completed"
    turn2 = planner.seen[2]                                   # the call after the first fetch
    assert turn2[-1]["role"] == "tool" and "Full output of step 2 (executor)" in turn2[-1]["content"]
    assert "output of step 2" in turn2[-1]["content"]
    assert "There is no step 99" in planner.seen[3][-1]["content"]
    notes = [c.note for c in traj.calls if c.role == "planner"]
    assert notes.count("fetch:2") == 1 and notes.count("fetch:99") == 1


def test_notes_view_missing_notes_keeps_old_ones():
    env = FakeEnv()
    planner = ScriptedLLM("planner", [
        ToolReply("p1 = 1", text="", instructions=INSTR, notes="KEEP-ME"),
        ToolReply("p2 = 1", text="", instructions=INSTR),         # forgot progress_notes
        ToolReply("apis.supervisor.complete_task()", text="", instructions="", notes="done"),
    ])
    executor = ScriptedLLM("executor", ["HANDBACK: ok", "HANDBACK: ok"])
    _, traj, _ = run_score(env, planner, executor, NCFG, verbose=False)
    assert "KEEP-ME" in planner.seen[2][0]["content"]
    assert any("no_notes" in c.note for c in traj.calls if c.role == "planner")


def test_full_view_unchanged_by_default():
    env = FakeEnv()
    planner = ScriptedLLM("planner", [tplan("a=1"), ToolReply("apis.supervisor.complete_task()")])
    executor = ScriptedLLM("executor", ["HANDBACK: done"])
    run_score(env, planner, executor, CFG, verbose=False)
    assert len(planner.seen[1]) > 1   # one growing conversation
