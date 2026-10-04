"""The orchestrator-built context: AppWorld reference parsing, state summaries, and what the executor sees."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score.context import SharedContext  # noqa: E402
from score.env import appworld_reference_from_step, summarize_namespace  # noqa: E402

# Shapes taken from the real AppWorld runs
CREATE_TX_DOC = json.dumps({
    "app_name": "venmo", "api_name": "create_transaction", "path": "/transactions", "method": "POST",
    "description": "Send money to a user.",
    "parameters": [
        {"name": "receiver_email", "type": "string", "required": True, "default": None, "constraints": ["value is email address"]},
        {"name": "amount", "type": "number", "required": True, "default": None, "constraints": ["value > 0.0"]},
        {"name": "access_token", "type": "string", "required": True, "default": None, "constraints": []},
        {"name": "description", "type": "string", "required": False, "default": "", "constraints": []},
    ],
    "response_schemas": {"success": {"message": "string", "transaction_id": 1}, "failure": {"message": "string"}},
}, indent=1)


def test_api_doc_becomes_one_line_signature():
    [(key, text)] = appworld_reference_from_step(
        "print(apis.api_docs.show_api_doc(app_name='venmo', api_name='create_transaction'))", CREATE_TX_DOC)
    assert key == "doc:venmo.create_transaction"
    assert text.startswith("venmo.create_transaction(receiver_email: string [value is email address], amount: number")
    assert "description: string = ''" in text and "Send money to a user." in text
    assert text.endswith("Returns {message, transaction_id}")


def test_api_descriptions_and_app_list():
    out = json.dumps([{"name": "login", "description": "Login."}, {"name": "search_contacts", "description": "x"}])
    assert appworld_reference_from_step("print(apis.api_docs.show_api_descriptions(app_name='phone'))", out) == \
        [("apis:phone", "phone APIs: login, search_contacts")]
    # the planner's labelled python-repr style from the real run
    out2 = "Venmo API names: ['show_account', 'create_transaction']"
    code2 = "venmo_api_descs = apis.api_docs.show_api_descriptions(app_name='venmo')\nprint('Venmo API names:', [...])"
    assert appworld_reference_from_step(code2, out2) == [("apis:venmo", "venmo APIs: show_account, create_transaction")]
    apps = json.dumps([{"name": "venmo", "description": "pay"}, {"name": "phone", "description": "call"}])
    assert appworld_reference_from_step("print(apis.api_docs.show_app_descriptions())", apps) == [("apps", "Apps: venmo, phone")]


def test_non_doc_steps_and_errors_are_ignored():
    assert appworld_reference_from_step("x = apis.venmo.login(...)", '{"access_token": "t"}') == []
    assert appworld_reference_from_step("print(apis.api_docs.show_api_doc(app_name='a', api_name='b'))",
                                        "Execution failed. Traceback: ...") == []


def test_namespace_summary_skips_setup_and_functions():
    import json as json_mod

    ns = {"json": json_mod, "apis": object(), "print": print, "In": [], "_": 1,
          "phone_access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9." + "x" * 200,
          "kristin": {"name": "Kristin", "phone": "6017026518"}, "helper": lambda: 1}
    text = summarize_namespace(ns, baseline={"json", "print", "In"})
    lines = text.splitlines()
    assert [ln.split(":")[0] for ln in lines] == ["phone_access_token", "kristin"]
    assert lines[0].startswith("phone_access_token: str = 'eyJ") and lines[0].endswith("…")
    assert "kristin: dict (len 2)" in lines[1]
    assert summarize_namespace({}, set()) == "(no variables yet)"


def test_shared_context_replaces_by_key_and_caps_size():
    class Env:
        def reference_from_step(self, code, output):
            return [(code, output)]

    sc = SharedContext(max_reference_chars=30)
    sc.record_step(Env(), "a", "first a")
    sc.record_step(Env(), "b", "bbbbbbbbbbbbbbbbbbbbbb")
    sc.record_step(Env(), "a", "second a")
    assert list(sc.reference) == ["b", "a"] and sc.reference["a"] == "second a"
    assert sc.reference_text() == "second a"            # oldest dropped to fit the cap
    assert SharedContext().reference_text() == "(nothing looked up yet)"
    assert "not available" in SharedContext.state_text(object())


def test_hooks_that_raise_never_break_a_run():
    class Bad:
        def reference_from_step(self, code, output):
            raise RuntimeError

        def state_summary(self):
            raise RuntimeError

    sc = SharedContext()
    sc.record_step(Bad(), "x", "y")
    assert sc.reference == {} and "could not" in sc.state_text(Bad())


# --------------------------------------------------------------------------- end to end with the agent loop
def test_executor_window_gets_state_reference_and_notes():
    from score.agents import run_score
    from test_mock_loop import CFG, INSTR, FakeEnv, ScriptedLLM, ToolReply

    class HookEnv(FakeEnv):
        def state_summary(self):
            return f"STATE after {self.steps} steps"

        def reference_from_step(self, code, output):
            return [("doc:" + code, "REF " + code)] if code.startswith("doc") else []

    env = HookEnv()
    cfg = {**CFG, "score": {**CFG["score"], "k": 2, "planner_view": "notes"}}
    planner = ScriptedLLM("planner", [
        ToolReply("doc_venmo = 1", text="", instructions=INSTR, notes="NOTE: amount is 54"),
        ToolReply("x = 2", text="", instructions=INSTR, notes="NOTE2"),
        ToolReply("apis.supervisor.complete_task()", text="", instructions="", notes="done"),
    ])
    executor = ScriptedLLM("executor", [ToolReply("doc_phone = 1"), "HANDBACK: ok", "HANDBACK: ok"])
    run_score(env, planner, executor, cfg, verbose=False)
    w1 = executor.seen[0][0]["content"]
    assert "STATE after 1 steps" in w1 and "REF doc_venmo = 1" in w1 and "NOTE: amount is 54" in w1
    w2 = executor.seen[2][0]["content"]                 # second window: reference from both windows persists
    assert "REF doc_venmo = 1" in w2 and "REF doc_phone = 1" in w2 and "NOTE2" in w2
    assert "STATE after 3 steps" in w2
    assert "STATE after" in planner.seen[1][0]["content"]   # the planner (notes view) sees state too
