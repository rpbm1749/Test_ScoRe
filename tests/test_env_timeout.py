"""The Windows execution timeout (works on any OS, so it can be tested here)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from score.env import run_with_timeout  # noqa: E402


def slow(seconds):
    end = time.time() + seconds
    while time.time() < end:  # busy loop, like a model's runaway `while True`
        pass
    return "finished"


def test_fast_call_returns_normally():
    assert run_with_timeout(slow, 0.05, seconds=2) == (False, "finished")


def test_slow_call_is_interrupted():
    t0 = time.time()
    timed_out, _ = run_with_timeout(slow, 30, seconds=0.5)
    assert timed_out and time.time() - t0 < 5


def test_interrupt_swallowed_by_callee_still_counts():
    def swallowing():
        try:
            slow(30)
        except KeyboardInterrupt:  # IPython catches it and returns an error message
            return "Execution failed. KeyboardInterrupt"

    timed_out, out = run_with_timeout(swallowing, seconds=0.5)
    assert timed_out


def test_no_stray_interrupt_after_cancel():
    run_with_timeout(slow, 0.05, seconds=0.3)
    time.sleep(0.6)  # a stray interrupt would raise KeyboardInterrupt here


# --------------------------------------------------------------------------- apis guard
def test_reassigned_apis_is_restored():
    from score.env import restore_protected

    real_apis, real_req = object(), object()
    ns = {"apis": ["oops, a list of API descriptions"], "requester": real_req}
    note = restore_protected(ns, {"apis": real_apis, "requester": real_req})
    assert ns["apis"] is real_apis and ns["requester"] is real_req
    assert "reassigned `apis`" in note and "requester" not in note
    assert restore_protected(ns, {"apis": real_apis, "requester": real_req}) == ""


def test_env_execute_appends_guard_note():
    from score.env import AppWorldEnv

    real_apis = object()

    class Shell:
        user_ns = {"apis": real_apis}

    class World:
        apis = real_apis
        requester = None
        shell = Shell()

        def execute(self, code):
            if code.startswith("apis ="):
                self.shell.user_ns["apis"] = [1, 2]
            return "out"

    env = AppWorldEnv.__new__(AppWorldEnv)  # skip AppWorld setup
    env.world, env.steps, env.max_steps = World(), 0, 10
    assert env.execute("x = 1") == "out"
    out = env.execute("apis = apis.api_docs.show_api_descriptions(app_name='venmo')")
    assert out.startswith("out") and "restored" in out
    assert env.world.shell.user_ns["apis"] is real_apis
