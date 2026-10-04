"""Tests for the API client's rate-limit handling, using a fake HTTP transport."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from score import llm as llm_mod  # noqa: E402
from score.llm import LLMClient, QuotaExhausted, RequestTooLarge  # noqa: E402

OK = {
    "choices": [{"message": {"content": "```python\nprint(1)\n```", "reasoning": "r"}}],
    "usage": {
        "prompt_tokens": 50, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 20},
        "completion_tokens_details": {"reasoning_tokens": 4},
        "queue_time": 0.01, "prompt_time": 0.02, "completion_time": 0.03, "total_time": 0.05,
    },
}


def client_with(responses):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return responses.pop(0)

    c = LLMClient("t", "https://example.test/v1", "m", api_key="k", reasoning_effort="low", verbose=False)
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    return c, calls


def test_ok_parses_usage(monkeypatch):
    c, calls = client_with([httpx.Response(200, json=OK)])
    r = c.chat([{"role": "user", "content": "hi"}])
    assert r.content.startswith("```python")
    assert (r.usage.prompt_tokens, r.usage.completion_tokens, r.usage.cached_tokens) == (50, 7, 20)
    assert r.usage.reasoning_tokens == 4
    assert (r.usage.queue_time_s, r.usage.prompt_time_s, r.usage.completion_time_s, r.usage.total_time_s) == (0.01, 0.02, 0.03, 0.05)
    body = calls[0].read().decode()
    assert '"reasoning_effort":"low"' in body.replace(" ", "")


def test_429_retries_then_succeeds(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: sleeps.append(s))
    c, calls = client_with([
        httpx.Response(429, headers={"retry-after": "3"}, json={"error": {"message": "Rate limit reached ... tokens per minute (TPM)"}}),
        httpx.Response(200, json=OK),
    ])
    c.chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 2 and any(abs(s - 3.5) < 0.01 for s in sleeps)


def test_daily_quota_raises(monkeypatch):
    c, _ = client_with([httpx.Response(429, json={"error": {"message": "Rate limit reached for model on tokens per day (TPD): Limit 200000"}})])
    with pytest.raises(QuotaExhausted):
        c.chat([{"role": "user", "content": "hi"}])


def test_long_retry_after_counts_as_quota(monkeypatch):
    c, _ = client_with([httpx.Response(429, headers={"retry-after": "3600"}, json={"error": {"message": "Rate limit reached"}})])
    with pytest.raises(QuotaExhausted):
        c.chat([{"role": "user", "content": "hi"}])


def test_request_too_large(monkeypatch):
    c, _ = client_with([httpx.Response(413, json={"error": {"message": "Request too large for model on tokens per minute (TPM)"}})])
    with pytest.raises(RequestTooLarge):
        c.chat([{"role": "user", "content": "hi"}])


def test_growing_conversation_only_budgets_new_tokens():
    c, _ = client_with([httpx.Response(200, json=OK)])
    first = [{"role": "user", "content": "x" * 40000}]
    assert c._estimate_uncached(first) > 9000
    c.chat(first)
    grown = first + [{"role": "assistant", "content": "ok"}, {"role": "user", "content": "y" * 400}]
    assert c._estimate_uncached(grown) < 200
    other = [{"role": "user", "content": "z" * 40000}]
    assert c._estimate_uncached(other) > 9000


def test_timing_from_x_groq_fallback():
    from score.llm import parse_usage

    u = parse_usage({"usage": {"prompt_tokens": 1}, "x_groq": {"usage": {"prompt_time": 0.5, "total_time": 0.7}}}, 1.0)
    assert u.prompt_time_s == 0.5 and u.total_time_s == 0.7 and u.queue_time_s == 0.0


# The exact error from the first real run on Groq.
REAL_FAILED = '{"name": "python", "arguments": print(apis.api_docs.show_app_descriptions())\n}'


def test_code_recovered_from_failed_generation():
    from score.parsing import code_from_failed_generation

    assert code_from_failed_generation(REAL_FAILED) == "print(apis.api_docs.show_app_descriptions())"
    assert code_from_failed_generation('{"name": "execute_python", "arguments": "{\\"code\\": \\"x = 1\\"}"}') == "x = 1"
    assert code_from_failed_generation("just text") is None


def test_tool_use_failed_is_recovered_without_another_call():
    err = {"error": {"message": "Tool choice is none, but model called a tool", "type": "invalid_request_error",
                     "code": "tool_use_failed", "failed_generation": REAL_FAILED}}
    c, calls = client_with([httpx.Response(400, json=err)])
    r = c.chat([{"role": "user", "content": "hi"}])
    assert len(calls) == 1
    assert "print(apis.api_docs.show_app_descriptions())" in r.content and "```python" in r.content
    assert r.usage.estimated_calls == 1 and r.usage.calls == 1 and r.usage.prompt_tokens > 0


def test_tools_are_sent_and_tool_calls_returned():
    tc = [{"id": "c1", "type": "function", "function": {"name": "execute_python", "arguments": '{"code": "x=1"}'}}]
    ok = {"choices": [{"message": {"content": "", "tool_calls": tc}}], "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
    c, calls = client_with([httpx.Response(200, json=ok)])
    from score.actions import TOOLS

    r = c.chat([{"role": "user", "content": "hi"}], tools=TOOLS)
    body = json.loads(calls[0].read())
    assert body["tools"][0]["function"]["name"] == "execute_python" and body["tool_choice"] == "auto"
    assert r.tool_calls == tc


def test_stray_quotes_are_repaired():
    from score.parsing import code_from_failed_generation, repair_code

    # what the real run produced: a trailing quote after the code
    assert repair_code('print(apis.api_docs.show_app_descriptions())"') == "print(apis.api_docs.show_app_descriptions())"
    multi = 'passwords = apis.supervisor.show_account_passwords()\nx = passwords[0]["password"]\n"'
    assert repair_code(multi) == 'passwords = apis.supervisor.show_account_passwords()\nx = passwords[0]["password"]'
    # a JSON string that lost its closing quote
    assert repair_code('"print(1)\\nprint(2)') == "print(1)\nprint(2)"
    # valid code is left alone, including code that legitimately ends with a quote character
    assert repair_code('s = "a"') == 's = "a"'
    # failed generations in a few shapes
    assert code_from_failed_generation('{"name": "python", "arguments": print(1)"\n}') == "print(1)"
    assert code_from_failed_generation('{"name": "python", "arguments": "print(1)"') == "print(1)"


def test_trim_keeps_prompt_and_recent_steps_with_tool_pairs():
    from score.llm import TRIM_NOTE, _conservative_tokens, trim_messages

    msgs = [{"role": "user", "content": "P" * 3000}]
    for i in range(10):
        msgs.append({"role": "assistant", "content": f"step {i}",
                     "tool_calls": [{"id": f"c{i}", "type": "function", "function": {"name": "execute_python", "arguments": "{}"}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": "O" * 1500})
    out, trimmed = trim_messages(msgs, 3000)
    assert trimmed and out[0] == msgs[0] and out[1]["content"] == TRIM_NOTE
    assert _conservative_tokens(out) <= 3000
    assert out[-1]["tool_call_id"] == "c9"                       # newest step kept
    for a, b in zip(out[2::2], out[3::2]):                        # every tool call still has its result
        assert a["role"] == "assistant" and b["tool_call_id"] == a["tool_calls"][0]["id"]
    same, t2 = trim_messages(msgs[:3], 3000)
    assert not t2 and same == msgs[:3]


def test_client_trims_and_counts():
    c, calls = client_with([httpx.Response(200, json=OK)])
    c.max_context_tokens = 500
    msgs = [{"role": "user", "content": "P" * 300}] + [
        m for i in range(20) for m in ({"role": "assistant", "content": "a" * 300}, {"role": "user", "content": "u" * 300})
    ]
    r = c.chat(msgs)
    sent = json.loads(calls[0].read())["messages"]
    assert len(sent) < len(msgs) and r.usage.trimmed_calls == 1


def test_cerebras_time_info_is_read():
    from score.llm import parse_usage

    u = parse_usage({"usage": {"prompt_tokens": 3}, "time_info": {"queue_time": 0.1, "prompt_time": 0.2, "completion_time": 0.3, "total_time": 0.6}}, 1.0)
    assert (u.queue_time_s, u.prompt_time_s, u.completion_time_s, u.total_time_s) == (0.1, 0.2, 0.3, 0.6)


def test_hourly_limit_waits_then_continues(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: sleeps.append(s))
    c, calls = client_with([
        httpx.Response(429, json={"message": "Tokens per hour limit exceeded - too many tokens processed."}),
        httpx.Response(200, json=OK),
    ])
    c.chat([{"role": "user", "content": "hi"}])
    assert 300 in sleeps and len(calls) == 2


def test_reasoning_is_passed_back_and_extra_body_sent():
    from score.actions import parse_action

    tc = [{"id": "c1", "type": "function", "function": {"name": "execute_python", "arguments": '{"code": "x=1"}'}}]
    ok = {"choices": [{"message": {"content": "", "reasoning_content": "I should set x.", "tool_calls": tc}}],
          "usage": {"prompt_tokens": 5, "completion_tokens": 3}}
    c, calls = client_with([httpx.Response(200, json=ok)])
    c.pass_back_reasoning = True
    c.extra_body = {"chat_template_kwargs": {"thinking": True}}
    r = c.chat([{"role": "user", "content": "hi"}])
    assert json.loads(calls[0].read())["chat_template_kwargs"] == {"thinking": True}
    act = parse_action(r)
    assert act.assistant_message["reasoning_content"] == "I should set x."
    assert r.reasoning == "I should set x."

    c2, _ = client_with([httpx.Response(200, json=ok)])  # off by default
    assert "reasoning_content" not in parse_action(c2.chat([{"role": "user", "content": "hi"}])).assistant_message


def test_probe_use_sets_planner_model(tmp_path, monkeypatch):
    import probe_models

    cfg = (Path(__file__).resolve().parents[1] / "config.yaml").read_text()
    (tmp_path / "config.yaml").write_text(cfg)
    monkeypatch.setattr(probe_models, "ROOT", tmp_path)
    probe_models.set_planner_model("moonshotai/kimi-k3")
    import yaml

    new = yaml.safe_load((tmp_path / "config.yaml").read_text())
    assert new["models"]["planner"]["model"] == "moonshotai/kimi-k3"
    assert new["models"]["executor"]["model"] == "openai/gpt-oss-20b"
