"""Tests for the API client's rate-limit handling, using a fake HTTP transport."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from score import llm as llm_mod  # noqa: E402
from score.llm import LLMClient, QuotaExhausted, RequestTooLarge  # noqa: E402

OK = {
    "choices": [{"message": {"content": "```python\nprint(1)\n```", "reasoning": "r"}}],
    "usage": {"prompt_tokens": 50, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 20}},
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
