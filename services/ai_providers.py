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
import threading
from typing import Dict, Any, Tuple, Optional

from openai import OpenAI
from dotenv import load_dotenv

from services.supabase_client import get_supabase_client

load_dotenv()

_OPENAI_KEY = os.getenv("OPENAI_API_KEY")
_GROQ_KEY = os.getenv("GROQ_API_KEY")
_GEMINI_KEY = os.getenv("GEMINI_API_KEY")

_openai_client = OpenAI(api_key=_OPENAI_KEY) if _OPENAI_KEY else None
_groq_client = (
    OpenAI(api_key=_GROQ_KEY, base_url="https://api.groq.com/openai/v1") if _GROQ_KEY else None
)
# Gemini through its OpenAI-compatible endpoint, so it is just another OpenAI
# client and every existing call site (response_format, temperature, max_tokens)
# keeps working unchanged. The free tier is the cheapest provider we have; its
# quotas are account-scoped and Google no longer publishes them, so a 429 must
# never be fatal -> chat_with_fallback() walks a chain rather than a single hop.
_gemini_client = (
    OpenAI(api_key=_GEMINI_KEY, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
    if _GEMINI_KEY else None
)

_GROQ_LLM = os.getenv("GROQ_LLM_MODEL", "llama-3.3-70b-versatile")
# Same env var as deck_ai_gemini.py on purpose: Google retires model names
# without notice, and one rename should fix every caller.
_GEMINI_LLM = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

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
        "providers": ["gemini", "groq", "openai"],
        "default": "gemini",  # bulk, non-user-facing -> the free tier is the right home for it
        "models": {"openai": "gpt-4o-mini", "groq": _GROQ_LLM, "gemini": _GEMINI_LLM},
    },
    "gd_brief": {
        "label": "GD brief generation",
        "providers": ["gemini", "openai", "groq"],
        # Owner directive 2026-09-22: the whole GD brief surface runs on the
        # Gemini free tier. OpenAI stays LAST in the chain purely as a safety
        # net, so a Gemini quota wall degrades cost rather than the page.
        "default": "gemini",
        "models": {"openai": "gpt-4o", "groq": _GROQ_LLM, "gemini": _GEMINI_LLM},
    },
    "daily_content": {
        "label": "Daily content generation",
        "providers": ["gemini", "openai", "groq"],
        "default": "openai",
        "models": {"openai": "gpt-4o", "groq": _GROQ_LLM, "gemini": _GEMINI_LLM},
    },
    "abstract_brief": {
        "label": "Abstract GD brief generation",
        "providers": ["gemini", "openai", "groq"],
        "default": "gemini",  # owner directive 2026-09-22: GD surface on the free tier
        "models": {"openai": "gpt-4o", "groq": _GROQ_LLM, "gemini": _GEMINI_LLM},
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
    "seo_writer": {
        "label": "Growth: SEO article generation",
        "providers": ["openai", "groq"],
        "default": "openai",  # public-facing quality — gpt-4o default; toggle to Groq to A/B cost
        "models": {"openai": "gpt-4o", "groq": _GROQ_LLM},
    },
    "seo_critique": {
        "label": "Growth: SEO quality critique (self-QA)",
        "providers": ["groq", "openai"],
        "default": "groq",  # cheap classifier-style pass; Groq is plenty
        "models": {"openai": "gpt-4o-mini", "groq": _GROQ_LLM},
    },
    "coach_planner": {
        "label": "Prep Copilot planner (tool orchestration)",
        "providers": ["openai", "groq"],
        "default": "openai",  # gpt-4o-mini — choosing which specialist to deploy must NEVER pay gpt-4o
        "models": {"openai": "gpt-4o-mini", "groq": _GROQ_LLM},
    },
    "coach_synthesis": {
        "label": "Prep Copilot final plan writing (quality)",
        "providers": ["openai", "groq"],
        "default": "openai",  # gpt-4o — this is the prose the candidate READS; keep it premium
        "models": {"openai": "gpt-4o", "groq": _GROQ_LLM},
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


# Gemini's OpenAI-compatibility layer does not document support for
# `response_format={"type": "json_object"}`. Sending it risks either a 400 or —
# worse — a 200 whose body is a ```json fenced block, which fails at parse time
# where the provider chain can no longer help. So for Gemini we drop the param
# and ask for raw JSON in words instead; callers parse with
# services.model_json.parse_model_json, which tolerates fences either way.
_JSON_ONLY_NUDGE = (
    "Return ONLY a single raw JSON object. No markdown code fences, no commentary "
    "before or after, no trailing commas."
)


def _kwargs_for(provider: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Per-provider adjustment of the call kwargs. Never mutates the caller's dict."""
    if provider != "gemini":
        return kwargs
    rf = kwargs.get("response_format")
    if not (isinstance(rf, dict) and rf.get("type") == "json_object"):
        return kwargs
    adjusted = dict(kwargs)
    adjusted.pop("response_format", None)
    # Headroom. Without the json_object constraint the model is freer to open with
    # a sentence or a ```json fence, and those tokens come out of max_tokens — a
    # budget tuned for a terse OpenAI reply can truncate mid-object, which the
    # tolerant parser cannot repair. Free tier, so the extra tokens cost nothing.
    if isinstance(adjusted.get("max_tokens"), int):
        adjusted["max_tokens"] = int(adjusted["max_tokens"] * 1.5)
    messages = list(adjusted.get("messages") or [])
    if messages and messages[0].get("role") == "system":
        head = dict(messages[0])
        head["content"] = f"{head.get('content', '')}\n\n{_JSON_ONLY_NUDGE}"
        messages[0] = head
    else:
        messages.insert(0, {"role": "system", "content": _JSON_ONLY_NUDGE})
    adjusted["messages"] = messages
    return adjusted


# Free-tier Gemini is rate-limited per minute, and the quota is account-scoped
# and unpublished. The news classifier fires several chunk calls back to back, so
# without spacing the later chunks 429 and fall through to a PAID provider —
# turning a rate limit into a bill. Same env var as deck_ai_gemini.py so one
# setting governs every Gemini caller; set 0 to disable.
# ...but ONLY for background jobs. Spacing an interactive call would queue users
# behind each other — a candidate waiting 30s for a brief is a worse outcome than
# a rare 429 falling through to a paid provider for a few cents. Bulk jobs have
# nobody waiting, so there the wait is free and the saving is real.
_BULK_FEATURES = {"news_classify", "daily_content", "seo_writer", "seo_critique"}

_gemini_last_call = 0.0
_gemini_gate = threading.Lock()


def _throttle_gemini() -> None:
    global _gemini_last_call
    try:
        min_gap = float(os.getenv("GEMINI_MIN_INTERVAL_SEC", "4.5"))
    except ValueError:
        min_gap = 4.5
    if min_gap <= 0:
        return
    with _gemini_gate:
        wait = min_gap - (time.time() - _gemini_last_call)
        if wait > 0:
            time.sleep(wait)
        _gemini_last_call = time.time()


def _client_for(provider: str) -> Optional[OpenAI]:
    """The configured client for a provider name, or None when its key is unset."""
    return {"openai": _openai_client, "groq": _groq_client, "gemini": _gemini_client}.get(provider)


def _available(feature: str, provider: str) -> bool:
    spec = FEATURES.get(feature) or {}
    return provider in spec.get("providers", []) and _client_for(provider) is not None


def current_provider(feature: str) -> str:
    """The provider serving `feature` right now, honoring the admin override, the
    code default, the feature's allow-list, and whether that provider's key is
    actually set. Falls through the feature's own provider order, so a missing
    GEMINI_API_KEY or GROQ_API_KEY degrades quietly instead of 500ing."""
    spec = FEATURES.get(feature)
    if not spec:
        return "openai"
    provider = _load_settings().get(feature, spec["default"])
    if provider not in spec["providers"]:
        provider = spec["default"]
    if _client_for(provider) is not None:
        return provider
    for candidate in spec["providers"]:
        if _client_for(candidate) is not None:
            return candidate
    return spec["default"]


def resolve_llm(feature: str) -> Tuple[Optional[OpenAI], str, str]:
    """Return (client, model, provider) for an LLM feature."""
    spec = FEATURES[feature]
    provider = current_provider(feature)
    return _client_for(provider), spec["models"].get(provider, spec["models"].get("openai")), provider


def openai_client() -> Optional[OpenAI]:
    return _openai_client


def chat_with_fallback(feature: str, **kwargs) -> Tuple[Any, str, str]:
    """Run a NON-streaming chat completion for `feature`, walking a CHAIN of
    providers rather than a single fallback.

    Order: the selected provider first, then the rest of the feature's allow-list
    in declared order, with OpenAI always tried last when it is allowed. Any error
    moves to the next link — a Gemini free-tier 429, a Groq JSON-mode quirk, a
    retired model name. This matters because the cheap providers are the ones with
    quotas we do not control: without the chain, a quota wall at 6 a.m. empties a
    user-facing page.

    Returns (response, model_used, provider_used). `model` must NOT be in kwargs;
    it is chosen here."""
    spec = FEATURES[feature]
    first = current_provider(feature)

    # The fallback order IS the feature's `providers` list, read left to right,
    # with whatever is currently selected tried first. Keeping it declarative
    # means the chain is legible from the FEATURES table above instead of being
    # encoded here — put the provider you want as the safety net LAST in that
    # list. (An earlier version forced OpenAI to the end universally, which
    # quietly sent a failed GD brief to Groq's Llama before gpt-4o ever saw it.)
    chain = [first] + [p for p in spec["providers"] if p != first]

    last_error: Optional[Exception] = None
    for provider in chain:
        cli = _client_for(provider)
        if cli is None:
            continue
        model = spec["models"].get(provider)
        if not model:
            continue
        try:
            if provider != first:
                print(f"[ai_providers] {feature}: falling through to {provider} ({model})")
            if provider == "gemini" and feature in _BULK_FEATURES:
                _throttle_gemini()
            return cli.chat.completions.create(model=model, **_kwargs_for(provider, kwargs)), model, provider
        except Exception as e:  # noqa: BLE001 — any failure is a reason to try the next link
            last_error = e
            print(f"[ai_providers] {provider} failed for {feature}: {type(e).__name__}: {e}")
            continue

    if last_error is not None:
        raise last_error
    raise RuntimeError(f"No provider available for {feature}: none of {chain} has a configured key")


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
            "gemini_available": _gemini_client is not None,
            "available": {p: _client_for(p) is not None for p in spec["providers"]},
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
