"""The pre-execution check and catalog lookups, using the actual mistakes from the AppWorld runs."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from score.env import (  # noqa: E402
    ApiCatalog,
    catalog_reference_for_code,
    catalog_reference_for_text,
    validate_appworld_code,
)


def P(name, typ="string", required=True, default=None):
    return {"name": name, "type": typ, "required": required, "default": default, "constraints": []}


# A small catalog in AppWorld's shape (app -> api -> doc)
CATALOG = ApiCatalog({
    "api_docs": {
        "show_app_descriptions": {"description": "Apps.", "parameters": []},
        "show_api_descriptions": {"description": "APIs of an app.", "parameters": [P("app_name")]},
        "show_api_doc": {"description": "Doc of an API.", "parameters": [P("app_name"), P("api_name")]},
    },
    "supervisor": {"show_account_passwords": {"description": "Passwords.", "parameters": []}},
    "phone": {
        "login": {"description": "Login.", "parameters": [P("username"), P("password")],
                  "response_schemas": {"success": {"access_token": "s", "token_type": "s"}}},
        "search_contacts": {"description": "Search contacts.", "parameters": [
            P("access_token"), P("query", required=False, default=""), P("page_index", "integer", False, 0)],
            "response_schemas": {"success": [{"contact_id": 1, "first_name": "s", "phone_number": "s"}]}},
        "search_text_messages": {"description": "Search texts.", "parameters": [P("access_token"), P("phone_number", required=False)]},
    },
    "venmo": {"create_transaction": {"description": "Send money to a user.", "parameters": [
        P("receiver_email"), P("amount", "number"), P("access_token"), P("description", required=False, default="")]}},
})
SESSION = {"apis", "print", "json", "passwords", "phone_login2"}


def check(code, existing=SESSION):
    return validate_appworld_code(code, CATALOG, set(existing))


# ---- the actual mistakes from score_k3_notes_v2 ----------------------------------------------
def test_wrong_parameter_name_and_missing_token():                       # step 3
    msg = check("contacts = apis.phone.search_contacts(name='Kristin')\nprint(contacts)")
    assert "has no parameter(s) name" in msg and "missing required parameter(s) access_token" in msg
    assert "phone.search_contacts(access_token: string, query: string = ''" in msg   # shows the real signature


def test_unquoted_email_is_caught():                                       # step 5
    msg = check("phone_login = apis.phone.login(username=matthew.blac@gmail.com, password=passwords[0]['password'])")
    assert "Not defined: gmail, matthew" in msg


def test_variable_that_does_not_exist():                                    # step 8
    msg = check("contacts = apis.phone.search_contacts(access_token=phone_access_token, query='Kristin')")
    assert "Not defined: phone_access_token" in msg and "phone_login2" in msg


def test_doc_lookup_for_an_api_that_does_not_exist():                      # step 9
    msg = check("print(apis.api_docs.show_api_doc(app_name='phone', api_name='get_text_messages'))")
    assert "The phone app has no API `get_text_messages`" in msg and "search_text_messages" in msg


# ---- other checks ------------------------------------------------------------------------------
def test_unknown_app_and_api():
    assert "There is no app `paypal`" in check("apis.paypal.send(x=1)")
    assert "The venmo app has no API `send_money`" in check("apis.venmo.send_money(amount=5)")


def test_valid_code_passes():
    code = """
token = apis.phone.login(username='4886643554', password=passwords[0]['password'])['access_token']
found = []
for page in range(3):
    res = apis.phone.search_contacts(access_token=token, query='Kristin', page_index=page)
    found += [c for c in res if c['first_name'] == 'Kristin']
try:
    x = {k: v for k, v in enumerate(found)}
except Exception as e:
    print(e)
def helper(a, *args, **kw):
    return a
import math as m
print(json.dumps(found), m.pi, helper(1), len(found), f"{token[:4]}")
"""
    assert check(code) is None


def test_kwargs_spread_skips_missing_check_but_syntax_errors_are_caught():
    assert check("args = {'access_token': 't'}\napis.phone.search_contacts(**args)") is None
    assert check("print('unterminated)").startswith("SyntaxError")


def test_without_catalog_only_names_and_syntax_are_checked():
    assert validate_appworld_code("apis.anything.goes(x=1)", None, {"apis"}) is None
    assert "Not defined: y" in validate_appworld_code("print(y)", None, {"print"})


def test_catalog_reference_for_code_and_text():
    refs = dict(catalog_reference_for_code(
        "t = apis.phone.login(username='u', password='p')\n"
        "print(apis.api_docs.show_api_doc(app_name='venmo', api_name='create_transaction'))\n"
        "print(apis.api_docs.show_api_descriptions(app_name='phone'))", CATALOG))
    assert set(refs) == {"doc:phone.login", "doc:venmo.create_transaction", "apis:phone"}
    assert refs["apis:phone"] == "phone APIs: login, search_contacts, search_text_messages"
    text = "SUBGOAL: pay\nSTEPS:\n1. Use phone.search_text_messages with phone_access_token\n2. venmo.create_transaction(...)\n3. ignore foo.bar"
    keys = [k for k, _ in catalog_reference_for_text(text, CATALOG)]
    assert keys == ["doc:phone.search_text_messages", "doc:venmo.create_transaction"]


def test_catalog_accepts_munch_like_mappings():
    class Munch(dict):
        pass

    cat = ApiCatalog(Munch(phone=Munch(login=Munch(description="Login.", parameters=[P("username")]))))
    assert cat.doc("phone", "login")["api_name"] == "login" and cat.apis("phone") == ["login"]


# ---- in the agent loops ----------------------------------------------------------------------------
class CheckingEnv:
    """FakeEnv plus a validate hook that rejects code containing 'BAD'."""

    def __new__(cls):
        from test_mock_loop import FakeEnv

        class _E(FakeEnv):
            def validate(self, code):
                return "- uses BAD" if "BAD" in code else None

        return _E()


def test_rejected_steps_dont_run_or_use_a_step():
    from score.agents import run_single
    from test_mock_loop import CFG, ScriptedLLM, ToolReply

    env = CheckingEnv()
    llm = ScriptedLLM("agent", [ToolReply("BAD = 1"), ToolReply("ok = 1"), ToolReply("apis.supervisor.complete_task()")])
    reason, traj, _ = run_single(env, llm, CFG, role="executor", verbose=False)
    assert reason == "completed" and env.codes == ["ok = 1", "apis.supervisor.complete_task()"] and env.steps == 2
    assert len(traj.rejections) == 1 and "rejected" in traj.calls[0].note
    assert "Not executed" in llm.seen[1][-1]["content"] and "uses BAD" in llm.seen[1][-1]["content"]


def test_after_three_rejections_in_a_row_code_runs_anyway():
    from score.agents import run_single
    from test_mock_loop import CFG, ScriptedLLM, ToolReply

    env = CheckingEnv()
    llm = ScriptedLLM("agent", [ToolReply("BAD") for _ in range(4)] + [ToolReply("apis.supervisor.complete_task()")])
    run_single(env, llm, CFG, role="frontier", verbose=False)
    assert env.codes[0] == "BAD" and len([c for c in env.codes if c == "BAD"]) == 1


def test_planner_and_executor_are_checked_in_score():
    from score.agents import run_score
    from test_mock_loop import CFG, INSTR, ScriptedLLM, ToolReply

    env = CheckingEnv()
    cfg = {**CFG, "score": {**CFG["score"], "k": 2, "planner_view": "notes"}}
    planner = ScriptedLLM("planner", [
        ToolReply("BAD_planner", text="", instructions=INSTR, notes="n"),
        ToolReply("p = 1", text="", notes="n2"),                  # retry keeps the earlier instructions
        ToolReply("apis.supervisor.complete_task()", text="", instructions="", notes="done"),
    ])
    executor = ScriptedLLM("executor", [ToolReply("BAD_exec"), ToolReply("e = 1"), "HANDBACK: done"])
    reason, traj, extra = run_score(env, planner, executor, cfg, verbose=False)
    assert reason == "completed"
    assert "BAD_planner" not in env.codes and "BAD_exec" not in env.codes
    assert [r["role"] for r in traj.rejections] == ["planner", "executor"]
    assert extra["windows"][0]["instructions"] == INSTR
