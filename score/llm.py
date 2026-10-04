"""OpenAI-compatible chat client used for both Groq (hosted) and Ollama (local).

Handles the things that matter on a free tier:
  * client-side throttling to stay under requests/min and tokens/min,
  * retrying on 429 using the server's retry-after hint,
  * stopping cleanly when a *daily* quota is exhausted (QuotaExhausted),
  * flagging requests that are bigger than the per-minute token limit (RequestTooLarge),
  * recording token usage (prompt / completion / cached) for every call.
"""

from __future__ import annotations

import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import httpx


class QuotaExhausted(Exception):
    """Daily request/token quota used up. The runner stops and you resume later."""


class RequestTooLarge(Exception):
    """A single request is larger than the provider allows (e.g. Groq free-tier TPM)."""


class LLMError(Exception):
    """Any other non-recoverable API error."""


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    calls: int = 0
    latency_s: float = 0.0

    def add(self, other: "Usage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cached_tokens += other.cached_tokens
        self.calls += other.calls
        self.latency_s += other.latency_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "latency_s": round(self.latency_s, 2),
        }


@dataclass
class ChatResult:
    content: str
    reasoning: str
    usage: Usage
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


def _approx_tokens(messages: list[dict[str, str]]) -> int:
    # ~4 characters per token is close enough for throttling decisions.
    return sum(len(m.get("content") or "") for m in messages) // 4 + 8 * len(messages)


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
        self.rpm = rpm
        self.tpm = tpm
        self.max_retries = max_retries
        self.verbose = verbose
        self._http = httpx.Client(timeout=timeout_s)
        self._req_times: deque[float] = deque()
        self._tok_times: deque[tuple[float, int]] = deque()

    # ------------------------------------------------------------------ throttling
    def _throttle(self, est_tokens: int) -> None:
        while True:
            now = time.time()
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
        now = time.time()
        self._req_times.append(now)
        self._tok_times.append((now, tokens))

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"    [{self.name}] {msg}", flush=True)

    # ------------------------------------------------------------------ main call
    def chat(self, messages: list[dict[str, str]]) -> ChatResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_completion_tokens": self.max_tokens,
        }
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:
            body["temperature"] = self.temperature
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        est = _approx_tokens(messages) + min(self.max_tokens, 1024)
        backoff = 5.0
        for attempt in range(self.max_retries + 1):
            self._throttle(est)
            t0 = time.time()
            try:
                resp = self._http.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
            except httpx.HTTPError as e:
                if attempt == self.max_retries:
                    raise LLMError(f"{self.name}: network error {e!r}") from e
                self._log(f"network error ({e.__class__.__name__}), retrying in {backoff:.0f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
                continue
            latency = time.time() - t0

            if resp.status_code == 200:
                data = resp.json()
                msg = data["choices"][0]["message"]
                u = data.get("usage") or {}
                cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                usage = Usage(
                    prompt_tokens=u.get("prompt_tokens", 0),
                    completion_tokens=u.get("completion_tokens", 0),
                    cached_tokens=cached,
                    calls=1,
                    latency_s=latency,
                )
                self._record(max(usage.prompt_tokens - cached, 0) + usage.completion_tokens)
                return ChatResult(
                    content=msg.get("content") or "",
                    reasoning=msg.get("reasoning") or msg.get("reasoning_content") or "",
                    usage=usage,
                    raw=data,
                )

            text = resp.text
            lower = text.lower()
            self._record(0)
            if resp.status_code == 413 or "request too large" in lower or "reduce your message size" in lower:
                raise RequestTooLarge(f"{self.name}: {text[:400]}")
            if resp.status_code == 429:
                if re.search(r"per day|\(tpd\)|\(rpd\)|daily", lower):
                    raise QuotaExhausted(f"{self.name}: {text[:400]}")
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
        verbose=verbose,
    )
