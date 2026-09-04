"""
Central AI provider resolver — the one place that decides, per feature, whether a
call goes to OpenAI, Groq, or (for TTS) Google.

Why this exists: cost. Groq runs the same-quality Whisper ~9x cheaper and a fast
Llama for the interviewer; Google WaveNet TTS is ~3.75x cheaper than OpenAI and
more natural. But quality-sensitive jobs (scoring) must stay on OpenAI. So each
feature has a DEFAULT here, and an admin can override it live via
`ai_provider_settings` (surfaced in the admin panel) — no redeploy.

Design notes:
- `FEATURES` is the source of truth for what's toggleable and the model each
  provider uses. The admin UI is generated from it.
- Settings are read from the DB with a short in-memory cache (per worker), so the
  interviewer hot-path never hits the DB per turn and a toggle still propagates in
  ~CACHE_TTL seconds.
- Fails SAFE: if the DB read errors we keep the last cache / code defaults; if a
  feature is set to Groq but GROQ_API_KEY is unset, it transparently uses OpenAI.
  Scoring is LOCKED to OpenAI (its only allowed provider) so a toggle can never
  gamble the marking quality.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Any, Tuple, Optional

from openai import OpenAI
from dotenv import load_dotenv

from services.supabase_client import get_supabase_client

load_dotenv()

_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_GROQ_KEY = os.getenv("GROQ_API_KEY")

_openai_client = OpenAI(api_key=_OPENAI_KEY) if _OPENAI_KEY else None
_groq_client = (
    OpenAI(api_key=_GROQ_KEY, base_url="https://api.groq.com/openai/v1") if _GROQ_KEY else None
)

_GROQ_LLM = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")

# feature -> spec. `providers` is the ordered list the admin may choose from.
FEATURES: Dict[str, Dict[str, Any]] = {
    "stt": {
        "label": "Voice transcription (STT)",
        "providers": ["groq", "openai"],
        "default": "groq",
        "models": {"openai": "whisper-1", "groq": "whisper-large-v3-turbo"},
    },
    "interviewer": {
        "label": "Interviewer replies (typed + voice)",
        "providers": ["groq", "openai"],
        "default": "groq",
        "models": {"openai": "gpt-4o-mini", "groq": _GROQ_LLM},
    },
    "validity": {
        "label": "Gibberish / off-topic gate",
        "providers": ["groq", "openai"],
        "default": "groq",
        "models": {"openai": "gpt-4o-mini", "groq": _GROQ_LLM},
    },
    "news_classify": {
        "label": "News headline classification",
        "providers": ["groq", "openai"],
        "default": "groq",
        "models": {"openai": "gpt-4o-mini", "groq": _GROQ_LLM},
    },
    "gd_brief": {
        "label": "GD brief generation",
        "providers": ["openai", "groq"],
        "default": "openai",  # long structured output — gpt-4o safer; toggle to A/B
        "models": {"openai": "gpt-4o", "groq": _GROQ_LLM},
    },
    "daily_content": {
        "label": "Daily content generation",
        "providers": ["openai", "groq"],
        "default": "openai",
        "models": {"openai": "gpt-4o", "groq": _GROQ_LLM},
    },
    "abstract_brief": {
        "label": "Abstract GD brief generation",
        "providers": ["openai", "groq"],
        "default": "openai",  # user-facing brief — gpt-4o by default; toggle to A/B cost
        "models": {"openai": "gpt-4o", "groq": _GROQ_LLM},
    },
    "voice_mode": {
        "label": "Voice interview mode",
        "providers": ["pipeline", "realtime", "gemini"],
        "default": "pipeline",  # Groq pipeline (Groq STT + Groq LLM + Google TTS): cheap, no OpenAI. "realtime" = OpenAI speech-to-speech, kept as an option.
        "models": {"realtime": "gpt-realtime-2.1", "pipeline": "whisper + llm + tts", "gemini": "gemini-live"},
    },
    "scoring": {
        "label": "Answer scoring (locked — quality-critical)",
        "providers": ["openai"],  # intentionally NOT toggleable
        "default": "openai",
        "models": {"openai": "gpt-4o"},
    },
    "tts": {
        "label": "Interviewer voice (TTS)",
        "providers": ["openai", "google"],
        "default": "openai",
        "models": {"openai": "tts-1", "google": "wavenet"},
    },
}

# ---------------------------------------------------------------------------
# Cached settings
# ---------------------------------------------------------------------------
CACHE_TTL = 30.0
_cache: Dict[str, Any] = {"data": {}, "ts": 0.0}


def _load_settings(force: bool = False) -> Dict[str, str]:
    now = time.time()
    if not force and (now - _cache["ts"]) < CACHE_TTL and _cache["ts"] > 0:
        return _cache["data"]
    try:
        res = get_supabase_client().table("ai_provider_settings").select("feature, provider").execute()
        _cache["data"] = {r["feature"]: r["provider"] for r in (res.data or []) if r.get("feature")}
        _cache["ts"] = now
    except Exception:
        # keep whatever we last had (or {} -> code defaults). Never break a call.
        _cache["ts"] = now
    return _cache["data"]


def current_provider(feature: str) -> str:
    """The provider serving `feature` right now, honoring the admin override,
    the code default, availability of the Groq key, and the feature's allow-list."""
    spec = FEATURES.get(feature)
    if not spec:
        return "openai"
    provider = _load_settings().get(feature, spec["default"])
    if provider not in spec["providers"]:
        provider = spec["default"]
    if provider == "groq" and _groq_client is None:
        provider = "openai" if "openai" in spec["providers"] else spec["default"]
    return provider


def resolve_llm(feature: str) -> Tuple[OpenAI, str, str]:
    """Return (client, model, provider) for an LLM feature. Groq client is used
    only when selected AND available; otherwise OpenAI."""
    spec = FEATURES[feature]
    provider = current_provider(feature)
    if provider == "groq" and _groq_client is not None:
        return _groq_client, spec["models"]["groq"], "groq"
    return _openai_client, spec["models"]["openai"], "openai"


def openai_client() -> Optional[OpenAI]:
    return _openai_client


def chat_with_fallback(feature: str, **kwargs) -> Tuple[Any, str, str]:
    """Run a NON-streaming chat completion for `feature` on its selected provider,
    and if that provider (Groq) errors for ANY reason — including a model that
    handles response_format/JSON differently — transparently retry on OpenAI.
    Returns (response, model_used, provider_used). `model` must NOT be in kwargs;
    it is chosen here."""
    cli, model, provider = resolve_llm(feature)
    try:
        return cli.chat.completions.create(model=model, **kwargs), model, provider
    except Exception as e:
        if provider != "openai" and _openai_client is not None:
            oai_model = FEATURES[feature]["models"]["openai"]
            print(f"[ai_providers] {provider} failed for {feature} ({e}); falling back to OpenAI {oai_model}")
            return _openai_client.chat.completions.create(model=oai_model, **kwargs), oai_model, "openai"
        raise


# ---------------------------------------------------------------------------
# Admin surface
# ---------------------------------------------------------------------------
def list_features() -> list[dict]:
    """Everything the admin UI needs: current provider + the choices per feature."""
    settings = _load_settings(force=True)
    out = []
    for key, spec in FEATURES.items():
        chosen = settings.get(key, spec["default"])
        if chosen not in spec["providers"]:
            chosen = spec["default"]
        out.append({
            "feature": key,
            "label": spec["label"],
            "providers": spec["providers"],
            "default": spec["default"],
            "current": chosen,
            "model": spec["models"].get(chosen),
            "groq_available": _groq_client is not None,
        })
    return out


def set_provider(feature: str, provider: str, updated_by: Optional[str] = None) -> None:
    spec = FEATURES.get(feature)
    if not spec:
        raise ValueError(f"Unknown feature: {feature}")
    if provider not in spec["providers"]:
        raise ValueError(f"{provider} is not allowed for {feature}")
    get_supabase_client().table("ai_provider_settings").upsert(
        {"feature": feature, "provider": provider, "updated_by": updated_by},
        on_conflict="feature",
    ).execute()
    _load_settings(force=True)  # refresh cache immediately
