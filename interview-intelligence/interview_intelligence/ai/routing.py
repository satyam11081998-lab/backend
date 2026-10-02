"""Model routing: named routes -> ordered (provider, model) chains (spec §73, §95).

Routes: fast | strong | grounded | stt | tts. Per-prompt overrides are allowed:
  II_MODEL_ROUTES='{"strong":[{"provider":"anthropic","model":"claude-sonnet-4-5"},
                              {"provider":"openai","model":"gpt-4o"}],
                    "interviewer":[{"provider":"groq","model":"llama-3.3-70b-versatile"}]}'
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ..config import get_settings
from ..errors import Unavailable
from .provider import AnthropicProvider, InterviewAIProvider, OpenAICompatibleProvider


@dataclass(frozen=True)
class Target:
    provider: str
    model: str


_providers: Dict[str, InterviewAIProvider] = {}
_overrides: Dict[str, InterviewAIProvider] = {}
_lock = threading.Lock()


def simulation_allowed() -> bool:
    s = get_settings()
    if s.is_production:
        return False  # never, whatever the flags say
    return s.env in {"test", "dev"} or os.environ.get("II_ALLOW_SIMULATION", "").lower() in {"1", "true", "yes"}


def _build(name: str) -> Optional[InterviewAIProvider]:
    s = get_settings()
    if name == "openai" and s.openai_api_key:
        return OpenAICompatibleProvider("openai", s.openai_api_key, s.openai_base_url)
    if name == "groq" and s.groq_api_key:
        return OpenAICompatibleProvider("groq", s.groq_api_key, "https://api.groq.com/openai/v1")
    if name == "gemini" and s.gemini_api_key:
        return OpenAICompatibleProvider("gemini", s.gemini_api_key,
                                        "https://generativelanguage.googleapis.com/v1beta/openai",
                                        supports_json_mode=False)
    if name == "anthropic" and s.anthropic_api_key:
        return AnthropicProvider(s.anthropic_api_key)
    if name == "simulated" and simulation_allowed():
        from .simulated import SimulatedProvider
        return SimulatedProvider()
    return None


def get_provider(name: str) -> Optional[InterviewAIProvider]:
    if name in _overrides:
        return _overrides[name]
    with _lock:
        if name not in _providers:
            p = _build(name)
            if p is None:
                return None
            _providers[name] = p
        return _providers[name]


def set_provider_override(name: str, provider: Optional[InterviewAIProvider]) -> None:
    """Tests: force a provider (e.g. a failing one) under a name."""
    if provider is None:
        _overrides.pop(name, None)
    else:
        _overrides[name] = provider


def reset_for_tests() -> None:
    with _lock:
        _providers.clear()
    _overrides.clear()


def _defaults() -> Dict[str, List[Target]]:
    s = get_settings()
    fast = [Target(s.provider_fast, s.model_fast)]
    strong = [Target(s.provider_strong, s.model_strong)]
    if s.provider_fast != "openai":
        fast.append(Target("openai", "gpt-4o-mini"))
    if s.provider_strong != "openai":
        strong.append(Target("openai", "gpt-4o"))
    else:
        strong.append(Target("openai", "gpt-4o-mini"))  # degrade, don't die
    return {
        "fast": fast,
        "strong": strong,
        "grounded": [Target("gemini", "gemini-2.5-flash")],
        "stt": [Target("openai", "whisper-1"), Target("groq", "whisper-large-v3-turbo")],
        "tts": [Target("openai", "tts-1")],
    }


def _overrides_from_env() -> Dict[str, List[Target]]:
    raw = get_settings().model_routes_json
    if not raw:
        return {}
    try:
        data = json.loads(raw)
        return {k: [Target(str(t["provider"]), str(t["model"])) for t in v] for k, v in data.items()}
    except (ValueError, KeyError, TypeError):
        return {}


def resolve(prompt_id: str, route: str) -> List[Tuple[InterviewAIProvider, str]]:
    env = _overrides_from_env()
    chain = env.get(prompt_id) or env.get(route) or _defaults().get(route, [])
    out: List[Tuple[InterviewAIProvider, str]] = []
    for t in chain:
        p = get_provider(t.provider)
        if p is not None:
            out.append((p, t.model))
    if not out:
        sim = get_provider("simulated")
        if sim is not None:
            out.append((sim, "simulated"))
    if not out:
        raise Unavailable("The AI service is not configured. Please try again later.", code="ai_unconfigured")
    return out
