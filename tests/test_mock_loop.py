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


class ScriptedLLM:
    """Returns replies from a list; records every message list it was sent."""

    def __init__(self, name: str, replies: list[str]) -> None:
        self.name = name
        self.replies = list(replies)
        self.seen: list[list[dict]] = []

    def chat(self, messages):
        self.seen.append([dict(m) for m in messages])
        if not self.replies:
            raise AssertionError(f"{self.name} ran out of scripted replies")
        return ChatResult(content=self.replies.pop(0), reasoning="", usage=Usage(100, 10, 0, 1, 0.1))


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


def test_score_executor_completes_task():
    env = FakeEnv()
    planner = ScriptedLLM("planner", [plan("a=1")])
    executor = ScriptedLLM("executor", [code("apis.supervisor.complete_task(answer=1)")])
    reason, traj, _ = run_score(env, planner, executor, CFG, verbose=False)
    assert reason == "completed"
    assert traj.usage["planner"].calls == 1


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
