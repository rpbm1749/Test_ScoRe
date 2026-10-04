"""AppWorld freezes time with freezegun; our clock must keep running."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from score import clock  # noqa: E402

freezegun = pytest.importorskip("freezegun")


def test_clock_runs_while_time_is_frozen():
    with freezegun.freeze_time("2023-05-18 12:00:00"):
        assert time.time() == time.time()  # frozen, like inside an AppWorld task
        t0 = clock.now()
        time.sleep(0.05)
        assert clock.now() - t0 >= 0.04


def test_llm_latency_measured_while_frozen():
    import httpx

    from score.llm import LLMClient

    def handler(request):
        time.sleep(0.05)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hi"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    c = LLMClient("t", "https://example.test/v1", "m", api_key="k", verbose=False)
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    with freezegun.freeze_time("2023-05-18 12:00:00"):
        r = c.chat([{"role": "user", "content": "x"}])
    assert r.usage.latency_s >= 0.04
