"""OpenAI-compatible chat client used for both Groq (hosted) and Ollama (local).

Handles the things that matter on a free tier:
  * client-side throttling to stay under requests/min and tokens/min,
  * retrying on 429 using the server's retry-after hint,
  * stopping cleanly when a *daily* quota is exhausted (QuotaExhausted),
  * flagging requests that are bigger than the per-minute token limit (RequestTooLarge),
  * recording token usage (prompt / completion / cached) for every call.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx

from . import clock
from .parsing import code_from_failed_generation


class QuotaExhausted(Exception):
    """Daily request/token quota used up. The runner stops and you resume later."""


class RequestTooLarge(Exception):
    """A single request is larger than the provider allows (e.g. Groq free-tier TPM)."""


class LLMError(Exception):
    """Any other non-recoverable API error."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0          # includes reasoning tokens
    cached_tokens: int = 0
    reasoning_tokens: int = 0
    calls: int = 0
    estimated_calls: int = 0            # calls whose token counts had to be estimated (see chat())
    trimmed_calls: int = 0              # calls whose history was trimmed to fit max_context_tokens
    latency_s: float = 0.0              # measured on our side: request sent -> response received
    # Reported by Groq in each response's usage block (seconds). 0 if the provider doesn't send them.
    queue_time_s: float = 0.0           # waiting in Groq's queue
    prompt_time_s: float = 0.0          # hardware time processing the prompt
    completion_time_s: float = 0.0      # hardware time generating the output
    total_time_s: float = 0.0           # Groq's total for the request

    def add(self, other: "Usage") -> None:
        for f in (
            "prompt_tokens", "completion_tokens", "cached_tokens", "reasoning_tokens", "calls", "estimated_calls", "trimmed_calls",
            "latency_s", "queue_time_s", "prompt_time_s", "completion_time_s", "total_time_s",
        ):
            setattr(self, f, getattr(self, f) + getattr(other, f))

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "estimated_calls": self.estimated_calls,
            "trimmed_calls": self.trimmed_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "latency_s": round(self.latency_s, 3),
            "queue_time_s": round(self.queue_time_s, 4),
            "prompt_time_s": round(self.prompt_time_s, 4),
            "completion_time_s": round(self.completion_time_s, 4),
            "total_time_s": round(self.total_time_s, 4),
        }


def _num(x: Any) -> float:
    try:
        return float(x or 0)
    except (TypeError, ValueError):
        return 0.0


def parse_usage(data: dict[str, Any], latency: float) -> Usage:
    """Read token counts and the provider's timing fields (Groq or Cerebras) from a response."""
    u = data.get("usage") or {}
    # Groq has also sent timings under x_groq.usage in some API versions; use it as a fallback.
    xg = (data.get("x_groq") or {}).get("usage") or {}
    ti = data.get("time_info") or {}  # Cerebras reports its timings here

    def pick(key: str) -> float:
        return _num(u.get(key)) or _num(xg.get(key)) or _num(ti.get(key))

    return Usage(
        prompt_tokens=int(u.get("prompt_tokens", 0) or 0),
        completion_tokens=int(u.get("completion_tokens", 0) or 0),
        cached_tokens=int((u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0),
        reasoning_tokens=int((u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0),
        calls=1,
        latency_s=latency,
        queue_time_s=pick("queue_time"),
        prompt_time_s=pick("prompt_time"),
        completion_time_s=pick("completion_time"),
        total_time_s=pick("total_time"),
    )


@dataclass
class ChatResult:
    content: str
    reasoning: str
    usage: Usage
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    raw: dict[str, Any] = field(repr=False, default_factory=dict)
    # Extra fields to keep on the assistant message when it is sent back next turn
    # (e.g. reasoning_content, which some providers need for multi-turn tool use).
    assistant_extra: dict[str, Any] = field(default_factory=dict)


def _approx_tokens(messages: list[dict[str, str]]) -> int:
    # ~4 characters per token is close enough for throttling decisions.
    return sum(len(m.get("content") or "") for m in messages) // 4 + 8 * len(messages)


TRIM_NOTE = (
    "[Some earlier steps were removed from this conversation to fit the model's context limit. "
    "Variables created in those steps still exist in the REPL.]"
)


def _conservative_tokens(messages: list[dict[str, Any]]) -> int:
    """Over-estimate of prompt tokens (code and JSON tokenize at ~3 chars/token)."""
    n = 0
    for m in messages:
        n += len(m.get("content") or "")
        if m.get("tool_calls"):
            n += len(json.dumps(m["tool_calls"]))
    return n // 3 + 10 * len(messages)


def trim_messages(messages: list[dict[str, Any]], budget: int) -> tuple[list[dict[str, Any]], bool]:
    """Keep the first message (the task prompt) and as many of the most recent steps as fit.

    History is cut at assistant messages, so a tool call always stays with its tool result.
    If even the newest step doesn't fit, it is sent anyway and the provider decides."""
    if budget <= 0 or _conservative_tokens(messages) <= budget:
        return messages, False
    head, rest = messages[:1], messages[1:]
    segments: list[list[dict[str, Any]]] = []
    for m in rest:
        if m.get("role") == "assistant" or not segments:
            segments.append([])
        segments[-1].append(m)
    note = [{"role": "user", "content": TRIM_NOTE}]
    while len(segments) > 1 and _conservative_tokens(head + note + [m for seg in segments for m in seg]) > budget:
        segments.pop(0)
    return head + note + [m for seg in segments for m in seg], True


class LLMClient:
    def __init__(
        self,
        name: str,
        base_url: str,
        model: str,
        api_key: str | None = None,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 2048,
        rpm: int | None = None,
        tpm: int | None = None,
        max_context_tokens: int | None = None,
        extra_body: dict[str, Any] | None = None,
        pass_back_reasoning: bool = False,
        timeout_s: float = 600.0,
        max_retries: int = 8,
        verbose: bool = True,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_context_tokens = max_context_tokens
        self.extra_body = extra_body or {}
        self.pass_back_reasoning = pass_back_reasoning
        self.rpm = rpm
        self.tpm = tpm
        self.max_retries = max_retries
        self.verbose = verbose
        self._http = httpx.Client(timeout=timeout_s)
        self._req_times: deque[float] = deque()
        self._tok_times: deque[tuple[float, int]] = deque()
        # The previous successful prompt. When a new prompt starts with it (an agent conversation
        # that only grows), the provider can serve that prefix from its cache, and Groq does not
        # count cached tokens toward rate limits. So the throttle only budgets for the new part.
        self._last_messages: list[dict[str, str]] = []

    # ------------------------------------------------------------------ throttling
    def _throttle(self, est_tokens: int) -> None:
        while True:
            now = clock.now()
            while self._req_times and now - self._req_times[0] > 60:
                self._req_times.popleft()
            while self._tok_times and now - self._tok_times[0][0] > 60:
                self._tok_times.popleft()
            used = sum(t for _, t in self._tok_times)
            wait = 0.0
            if self.rpm and len(self._req_times) >= self.rpm:
                wait = max(wait, 60 - (now - self._req_times[0]) + 0.5)
            # Only wait on TPM if the request could fit in an empty window at all;
            # oversized requests are sent anyway (cached tokens may not count) and
            # the provider decides.
            if self.tpm and self._tok_times and used + est_tokens > self.tpm and est_tokens <= self.tpm:
                wait = max(wait, 60 - (now - self._tok_times[0][0]) + 0.5)
            if wait <= 0:
                return
            self._log(f"throttling {wait:.0f}s (rpm/tpm guard)")
            time.sleep(wait)

    def _record(self, tokens: int) -> None:
        now = clock.now()
        self._req_times.append(now)
        self._tok_times.append((now, tokens))

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"    [{self.name}] {msg}", flush=True)

    def _estimate_uncached(self, messages: list[dict[str, str]]) -> int:
        n = len(self._last_messages)
        if n and len(messages) > n and messages[:n] == self._last_messages:
            return _approx_tokens(messages[n:])
        return _approx_tokens(messages)

    def _recover_tool_failure(self, resp: httpx.Response, messages: list[dict[str, Any]], latency: float) -> ChatResult | None:
        """The model produced a tool call the provider couldn't parse (gpt-oss sometimes calls its
        built-in `python` tool with raw code). The error still contains what it generated, so we
        take the code from there instead of paying for another call. The provider sends no usage
        for failed calls, so token counts are estimated and the call is flagged."""
        failed = _failed_generation(resp)
        code = code_from_failed_generation(failed)
        if not code:
            return None
        prompt_est = _approx_tokens(messages)
        completion_est = max(len(failed) // 4, 1)
        usage = Usage(
            prompt_tokens=prompt_est, completion_tokens=completion_est, calls=1, estimated_calls=1, latency_s=latency
        )
        self._record(self._estimate_uncached(messages) + completion_est)
        self._last_messages = [dict(m) for m in messages]
        self._log("recovered code from a malformed tool call (token counts estimated)")
        return ChatResult(content=f"```python\n{code}\n```", reasoning="", usage=usage, tool_calls=[], raw={"recovered_from": failed})

    # ------------------------------------------------------------------ main call
    def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None) -> ChatResult:
        trimmed = False
        if self.max_context_tokens:
            messages, trimmed = trim_messages(messages, self.max_context_tokens)
        result = self._chat(messages, tools)
        if trimmed:
            result.usage.trimmed_calls = 1
        return result

    def _chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> ChatResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
        }
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:
            body["temperature"] = self.temperature
        body.update(self.extra_body)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        est = self._estimate_uncached(messages) + min(self.max_tokens, 1024)
        backoff = 5.0
        hourly_waits = 0
        for attempt in range(self.max_retries + 14):
            self._throttle(est)
            t0 = clock.now()
            try:
                resp = self._http.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
            except httpx.HTTPError as e:
                if attempt == self.max_retries:
                    raise LLMError(f"{self.name}: network error {e!r}") from e
                self._log(f"network error ({e.__class__.__name__}), retrying in {backoff:.0f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
                continue
            latency = clock.now() - t0

            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                usage = parse_usage(data, latency)
                self._record(max(usage.prompt_tokens - usage.cached_tokens, 0) + usage.completion_tokens)
                self._last_messages = [dict(m) for m in messages]
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                extra = {"reasoning_content": reasoning} if (self.pass_back_reasoning and reasoning) else {}
                return ChatResult(
                    content=msg.get("content") or "",
                    reasoning=reasoning,
                    usage=usage,
                    tool_calls=msg.get("tool_calls") or [],
                    raw=data,
                    assistant_extra=extra,
                )

            text = resp.text
            lower = text.lower()
            self._record(0)
            if resp.status_code == 400 and "tool_use_failed" in lower:
                recovered = self._recover_tool_failure(resp, messages, latency)
                if recovered is not None:
                    return recovered
            if resp.status_code == 413 or "request too large" in lower or "reduce your message size" in lower:
                raise RequestTooLarge(f"{self.name}: {text[:400]}")
            if resp.status_code == 429:
                if re.search(r"per day|\(tpd\)|\(rpd\)|daily", lower):
                    raise QuotaExhausted(f"{self.name}: {text[:400]}")
                if re.search(r"per hour|\(tph\)|hourly", lower) and hourly_waits < 13:
                    hourly_waits += 1
                    self._log("hourly token limit reached, waiting 5 min (up to ~1 h)")
                    time.sleep(300)
                    continue
                retry_after = resp.headers.get("retry-after")
                m = re.search(r"try again in ([\d.]+)(ms|s|m)", lower)
                if retry_after:
                    wait = float(retry_after)
                elif m:
                    val, unit = float(m.group(1)), m.group(2)
                    wait = val / 1000 if unit == "ms" else val * 60 if unit == "m" else val
                else:
                    wait = backoff
                    backoff = min(backoff * 2, 120)
                if attempt >= self.max_retries + hourly_waits:
                    raise LLMError(f"{self.name}: still rate limited after {attempt} retries: {text[:300]}")
                if wait > 300:
                    # A wait this long means a daily-style limit even if the message didn't say so.
                    raise QuotaExhausted(f"{self.name}: asked to wait {wait:.0f}s: {text[:400]}")
                self._log(f"429 rate limited, waiting {wait:.1f}s")
                time.sleep(wait + 0.5)
                continue
            if resp.status_code >= 500 and attempt < self.max_retries:
                self._log(f"server error {resp.status_code}, retrying in {backoff:.0f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
                continue
            raise LLMError(f"{self.name}: HTTP {resp.status_code}: {text[:400]}")
        raise LLMError(f"{self.name}: gave up after {self.max_retries} retries")


def _failed_generation(resp: httpx.Response) -> str:
    try:
        return ((resp.json().get("error") or {}).get("failed_generation")) or ""
    except ValueError:
        return ""


def build_client(role: str, model_cfg: dict[str, Any], providers: dict[str, Any], verbose: bool = True) -> LLMClient:
    provider = model_cfg["provider"]
    pcfg = providers[provider]
    api_key = None
    if pcfg.get("api_key_env"):
        api_key = os.environ.get(pcfg["api_key_env"])
        if not api_key:
            raise SystemExit(
                f"Missing {pcfg['api_key_env']}. Put it in a .env file next to config.yaml "
                f"(see .env.example)."
            )
    elif provider == "ollama":
        api_key = "ollama"
    return LLMClient(
        name=f"{role}:{model_cfg['model']}",
        base_url=pcfg["base_url"],
        model=model_cfg["model"],
        api_key=api_key,
        reasoning_effort=model_cfg.get("reasoning_effort"),
        temperature=model_cfg.get("temperature"),
        max_tokens=model_cfg.get("max_tokens", 2048),
        rpm=model_cfg.get("rpm"),
        tpm=model_cfg.get("tpm"),
        max_context_tokens=model_cfg.get("max_context_tokens"),
        extra_body=model_cfg.get("extra_body"),
        pass_back_reasoning=bool(model_cfg.get("pass_back_reasoning", False)),
        timeout_s=float(model_cfg.get("timeout_s", 600)),
        verbose=verbose,
    )
