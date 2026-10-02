"""InterviewAIProvider — the only way II talks to a model (spec §73).

Implementations speak plain HTTPS through httpx (no vendor SDKs: smaller memory
footprint, one code path, easy to mock):
* OpenAICompatibleProvider — OpenAI, Groq, Gemini (OpenAI-compat endpoint), any proxy.
* AnthropicProvider        — Claude Messages API.
* SimulatedProvider        — deterministic offline provider (tests, local dev without keys);
                             lives in ai/simulated.py.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Protocol

import httpx


@dataclass
class CompletionResult:
    text: str
    provider: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0


class ProviderError(RuntimeError):
    def __init__(self, message: str, *, retryable: bool = True, timeout: bool = False):
        super().__init__(message)
        self.retryable = retryable
        self.timeout = timeout


class InterviewAIProvider(Protocol):
    name: str

    def complete(self, messages: List[Dict[str, str]], *, model: str, temperature: float,
                 max_tokens: int, json_mode: bool, timeout_s: float,
                 meta: Optional[dict] = None) -> CompletionResult: ...

    def stream(self, messages: List[Dict[str, str]], *, model: str, temperature: float,
               max_tokens: int, timeout_s: float, meta: Optional[dict] = None) -> Iterator[str]: ...

    def transcribe(self, audio: bytes, *, filename: str, mime: str, model: str,
                   timeout_s: float) -> CompletionResult: ...

    def speak(self, text: str, *, model: str, voice: str, timeout_s: float) -> bytes: ...


def _raise_for(resp: httpx.Response, provider: str) -> None:
    if resp.status_code < 400:
        return
    retryable = resp.status_code in (408, 409, 425, 429) or resp.status_code >= 500
    body = resp.text[:300].replace("\n", " ")
    raise ProviderError(f"{provider} HTTP {resp.status_code}: {body}", retryable=retryable)


class OpenAICompatibleProvider:
    def __init__(self, name: str, api_key: str, base_url: str, *, supports_json_mode: bool = True,
                 client: Optional[httpx.Client] = None):
        self.name = name
        self._key = api_key
        self._base = base_url.rstrip("/")
        self._json_ok = supports_json_mode
        self._client = client or httpx.Client(timeout=60.0)

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self._key}", "Content-Type": "application/json"}

    def complete(self, messages, *, model, temperature, max_tokens, json_mode, timeout_s, meta=None):
        body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode and self._json_ok:
            body["response_format"] = {"type": "json_object"}
        t0 = time.perf_counter()
        try:
            resp = self._client.post(f"{self._base}/chat/completions", headers=self._headers(),
                                     json=body, timeout=timeout_s)
        except httpx.TimeoutException as e:
            raise ProviderError(f"{self.name} timeout", timeout=True) from e
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.name} transport error: {e}") from e
        _raise_for(resp, self.name)
        data = resp.json()
        usage = data.get("usage") or {}
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"{self.name}: malformed response") from e
        return CompletionResult(text=text, provider=self.name, model=data.get("model") or model,
                                input_tokens=int(usage.get("prompt_tokens") or 0),
                                output_tokens=int(usage.get("completion_tokens") or 0),
                                latency_ms=int((time.perf_counter() - t0) * 1000))

    def stream(self, messages, *, model, temperature, max_tokens, timeout_s, meta=None):
        body = {"model": model, "messages": messages, "temperature": temperature,
                "max_tokens": max_tokens, "stream": True}
        try:
            with self._client.stream("POST", f"{self._base}/chat/completions", headers=self._headers(),
                                     json=body, timeout=timeout_s) as resp:
                _raise_for(resp, self.name)
                for line in resp.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        delta = json.loads(chunk)["choices"][0]["delta"].get("content")
                    except (KeyError, IndexError, json.JSONDecodeError):
                        continue
                    if delta:
                        yield delta
        except httpx.TimeoutException as e:
            raise ProviderError(f"{self.name} stream timeout", timeout=True) from e
        except httpx.HTTPError as e:
            raise ProviderError(f"{self.name} stream error: {e}") from e

    def transcribe(self, audio, *, filename, mime, model, timeout_s):
        t0 = time.perf_counter()
        try:
            resp = self._client.post(
                f"{self._base}/audio/transcriptions",
                headers={"Authorization": f"Bearer {self._key}"},
                files={"file": (filename, audio, mime)},
                data={"model": model, "response_format": "json"},
                timeout=timeout_s,
            )
        except httpx.TimeoutException as e:
            raise ProviderError(f"{self.name} stt timeout", timeout=True) from e
        _raise_for(resp, self.name)
        text = (resp.json() or {}).get("text", "")
        return CompletionResult(text=text, provider=self.name, model=model,
                                latency_ms=int((time.perf_counter() - t0) * 1000))

    def speak(self, text, *, model, voice, timeout_s):
        try:
            resp = self._client.post(f"{self._base}/audio/speech", headers=self._headers(),
                                     json={"model": model, "voice": voice, "input": text, "format": "mp3"},
                                     timeout=timeout_s)
        except httpx.TimeoutException as e:
            raise ProviderError(f"{self.name} tts timeout", timeout=True) from e
        _raise_for(resp, self.name)
        return resp.content


class AnthropicProvider:
    API = "https://api.anthropic.com/v1/messages"

    def __init__(self, api_key: str, client: Optional[httpx.Client] = None):
        self.name = "anthropic"
        self._key = api_key
        self._client = client or httpx.Client(timeout=60.0)

    def _split(self, messages):
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        rest = [{"role": "assistant" if m["role"] == "assistant" else "user", "content": m["content"]}
                for m in messages if m["role"] != "system"]
        return system, rest

    def complete(self, messages, *, model, temperature, max_tokens, json_mode, timeout_s, meta=None):
        system, rest = self._split(messages)
        if json_mode:
            system += "\n\nRespond with a single JSON object and nothing else."
        t0 = time.perf_counter()
        try:
            resp = self._client.post(self.API, headers={
                "x-api-key": self._key, "anthropic-version": "2023-06-01", "content-type": "application/json",
            }, json={"model": model, "system": system, "messages": rest, "temperature": temperature,
                     "max_tokens": max_tokens}, timeout=timeout_s)
        except httpx.TimeoutException as e:
            raise ProviderError("anthropic timeout", timeout=True) from e
        except httpx.HTTPError as e:
            raise ProviderError(f"anthropic transport error: {e}") from e
        _raise_for(resp, self.name)
        data = resp.json()
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        usage = data.get("usage") or {}
        return CompletionResult(text=text, provider=self.name, model=data.get("model") or model,
                                input_tokens=int(usage.get("input_tokens") or 0),
                                output_tokens=int(usage.get("output_tokens") or 0),
                                latency_ms=int((time.perf_counter() - t0) * 1000))

    def stream(self, messages, *, model, temperature, max_tokens, timeout_s, meta=None):
        # Non-streaming fallback keeps one code path; the interviewer reply is short.
        yield self.complete(messages, model=model, temperature=temperature, max_tokens=max_tokens,
                            json_mode=False, timeout_s=timeout_s).text

    def transcribe(self, audio, *, filename, mime, model, timeout_s):
        raise ProviderError("anthropic has no speech-to-text endpoint", retryable=False)

    def speak(self, text, *, model, voice, timeout_s):
        raise ProviderError("anthropic has no text-to-speech endpoint", retryable=False)
