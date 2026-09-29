"""
Provider layer for the interviewer brain.

The brain never names a provider. This module resolves the client and model from
the existing admin toggle (ai_provider_settings feature 'interviewer'), with env
overrides, and performs at most ONE failover hop to OpenAI gpt-4o-mini. That hop
is a provider retry with the SAME prompt - not a different interviewer. If both
fail, ProviderError is raised and the turn fails visibly.

Every real model call is logged to ai_usage_log (so spend stays visible to the
daily-budget kill switch). Deterministic decisions never reach this module and
therefore never write a usage row.

All heavy imports are lazy so the decision layer imports on a bare interpreter.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

from services.interviewer.types import ProviderError, ProviderTimeout

FAILOVER_MODEL = "gpt-4o-mini"


@dataclass
class CallMeta:
    provider: str = ""
    model: str = ""
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None
    latency_ms: int = 0
    first_token_ms: Optional[int] = None
    failover: bool = False


def _timeout_default(channel: str) -> float:
    try:
        return float(os.getenv("INTERVIEWER_BRAIN_TIMEOUT_S", "12" if channel == "text" else "8"))
    except ValueError:
        return 10.0


def _resolve(feature: str = "interviewer") -> Tuple[Any, str, str]:
    from services.ai_providers import resolve_llm, openai_client, _client_for  # lazy
    forced_provider = (os.getenv("INTERVIEWER_BRAIN_PROVIDER") or "").strip().lower()
    forced_model = (os.getenv("INTERVIEWER_BRAIN_MODEL") or "").strip()
    if forced_provider:
        cli = _client_for(forced_provider)
        if cli is not None:
            return cli, forced_model or FAILOVER_MODEL, forced_provider
    cli, model, provider = resolve_llm(feature)
    if forced_model and provider == "openai":
        model = forced_model
    if cli is None:
        cli, model, provider = openai_client(), forced_model or FAILOVER_MODEL, "openai"
    return cli, model, provider


def _openai():
    from services.ai_providers import openai_client  # lazy
    return openai_client()


def _log(user_id: Optional[str], endpoint: str, model: str, resp: Any, latency_ms: int, success: bool, meta: Dict) -> None:
    try:
        from services.ai_usage import log_ai_usage  # lazy
        log_ai_usage(user_id=user_id, endpoint=endpoint, model=model, response=resp,
                     latency_ms=latency_ms, success=success, meta=meta)
    except Exception:
        pass


class _Usage:
    usage = None
    id = None


def _is_timeout(e: Exception) -> bool:
    name = type(e).__name__.lower()
    return "timeout" in name or "timed out" in str(e).lower()


class InterviewerLLM:
    """complete() and stream() with one failover hop. Instantiate per turn."""

    def __init__(self, *, user_id: Optional[str] = None, endpoint: str = "/attempts/messages",
                 channel: str = "text", temperature: float = 0.6):
        self.user_id = user_id
        self.endpoint = endpoint
        self.channel = channel
        self.temperature = temperature
        self.timeout = _timeout_default(channel)

    def _attempts(self):
        cli, model, provider = _resolve()
        yield cli, model, provider, False
        oa = _openai()
        if provider != "openai" and oa is not None:
            yield oa, FAILOVER_MODEL, "openai", True

    def complete(self, messages: List[Dict[str, str]], *, max_tokens: int) -> Tuple[str, CallMeta]:
        last: Optional[Exception] = None
        for cli, model, provider, failover in self._attempts():
            if cli is None:
                continue
            t0 = time.perf_counter()
            try:
                resp = cli.chat.completions.create(model=model, messages=messages, temperature=self.temperature,
                                                   max_tokens=max_tokens, timeout=self.timeout)
            except Exception as e:  # noqa: BLE001
                last = e
                _log(self.user_id, self.endpoint, model, None, int((time.perf_counter() - t0) * 1000), False,
                     {"brain": 1, "error": type(e).__name__, "provider": provider})
                continue
            ms = int((time.perf_counter() - t0) * 1000)
            _log(self.user_id, self.endpoint, model, resp, ms, True, {"brain": 1, "provider": provider})
            u = getattr(resp, "usage", None)
            text = ""
            try:
                text = resp.choices[0].message.content or ""
            except (AttributeError, IndexError):
                text = ""
            return text, CallMeta(provider=provider, model=model, latency_ms=ms, failover=failover,
                                  tokens_in=getattr(u, "prompt_tokens", None),
                                  tokens_out=getattr(u, "completion_tokens", None))
        if last is not None and _is_timeout(last):
            raise ProviderTimeout(f"interviewer model timed out: {type(last).__name__}")
        raise ProviderError(f"interviewer model unavailable: {type(last).__name__ if last else 'no client'}")

    def stream(self, messages: List[Dict[str, str]], *, max_tokens: int, meta: CallMeta) -> Iterator[str]:
        """Yields tokens. `meta` is filled in as the stream runs. A failure BEFORE the first token
        fails over once; a failure after tokens were yielded is raised (no silent retry)."""
        last: Optional[Exception] = None
        for cli, model, provider, failover in self._attempts():
            if cli is None:
                continue
            t0 = time.perf_counter()
            try:
                extra = {"stream_options": {"include_usage": True}} if provider == "openai" else {}
                stream = cli.chat.completions.create(model=model, messages=messages, temperature=self.temperature,
                                                     max_tokens=max_tokens, stream=True, timeout=self.timeout,
                                                     **extra)
            except Exception as e:  # noqa: BLE001
                last = e
                _log(self.user_id, self.endpoint, model, None, int((time.perf_counter() - t0) * 1000), False,
                     {"brain": 1, "error": type(e).__name__, "provider": provider})
                continue
            meta.provider, meta.model, meta.failover = provider, model, failover
            final = _Usage()
            yielded = False
            try:
                for chunk in stream:
                    if getattr(chunk, "usage", None):
                        final.usage = chunk.usage
                        final.id = getattr(chunk, "id", None)
                    try:
                        tok = chunk.choices[0].delta.content
                    except (AttributeError, IndexError):
                        tok = None
                    if tok:
                        if not yielded:
                            meta.first_token_ms = int((time.perf_counter() - t0) * 1000)
                        yielded = True
                        yield tok
            except Exception as e:  # noqa: BLE001
                meta.latency_ms = int((time.perf_counter() - t0) * 1000)
                _log(self.user_id, self.endpoint, model, final, meta.latency_ms, False,
                     {"brain": 1, "error": type(e).__name__, "provider": provider})
                if yielded:
                    raise ProviderError(f"stream interrupted: {type(e).__name__}")
                last = e
                continue
            meta.latency_ms = int((time.perf_counter() - t0) * 1000)
            u = final.usage
            meta.tokens_in = getattr(u, "prompt_tokens", None)
            meta.tokens_out = getattr(u, "completion_tokens", None)
            _log(self.user_id, self.endpoint, model, final, meta.latency_ms, True, {"brain": 1, "provider": provider})
            return
        if last is not None and _is_timeout(last):
            raise ProviderTimeout(f"interviewer model timed out: {type(last).__name__}")
        raise ProviderError(f"interviewer model unavailable: {type(last).__name__ if last else 'no client'}")


def json_completer(user_id: Optional[str] = None):
    """For the assessor: returns complete_json(messages, max_tokens, timeout) -> (text, meta)."""

    def complete_json(messages: List[Dict[str, str]], max_tokens: int, timeout: float):
        model = (os.getenv("INTERVIEWER_ASSESSOR_MODEL") or FAILOVER_MODEL).strip()
        cli = _openai()
        provider = "openai"
        if cli is None:
            cli, model, provider = _resolve()
        t0 = time.perf_counter()
        try:
            resp = cli.chat.completions.create(model=model, messages=messages, temperature=0.0, max_tokens=max_tokens,
                                               response_format={"type": "json_object"}, timeout=timeout)
        except Exception as e:  # noqa: BLE001
            _log(user_id, "/attempts/assess", model, None, int((time.perf_counter() - t0) * 1000), False,
                 {"brain": 1, "error": type(e).__name__})
            raise
        ms = int((time.perf_counter() - t0) * 1000)
        _log(user_id, "/attempts/assess", model, resp, ms, True, {"brain": 1})
        u = getattr(resp, "usage", None)
        return (resp.choices[0].message.content or "{}"), {
            "model": model, "provider": provider, "tokens_in": getattr(u, "prompt_tokens", None),
            "tokens_out": getattr(u, "completion_tokens", None)}

    return complete_json
