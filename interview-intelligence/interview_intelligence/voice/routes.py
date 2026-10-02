"""Voice (spec §29) — `voice.enabled` (default on) and `voice.engine`.

Two engines, one interview brain. Either way every interviewer line is decided by II's
orchestrator and every candidate answer goes through the normal /turns path, so scoring reads
the same transcript whether the candidate typed or talked.

* realtime (default): a live call. The browser opens a WebRTC connection straight to OpenAI
  Realtime with a short-lived client secret minted here (`POST /v1/voice/live`). The speech
  model is only the interviewer's VOICE: auto-replies are off, it speaks exactly the line the
  browser gives it, it hears the candidate continuously (barge-in, words-based end of turn) and
  transcribes them. The browser reports each response's token usage (`/live/usage`) so the
  spend is metered.
* standard: record -> `/transcribe` -> `/turns` -> `/speak`, sentence by sentence. Cheaper,
  slower, no talking over the interviewer. Also the fallback when a live call cannot connect.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Dict, Optional

import httpx
from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import Response
from pydantic import BaseModel

from ..access import flags, rate_limit
from ..access.plans import voice_engine_for
from ..ai.provider import ProviderError
from ..ai.routing import resolve
from ..ai.runner import record_live_usage, record_media_run, record_voice_minutes, spend_today_usd
from ..auth.assertion import Principal
from ..config import get_settings
from ..errors import Conflict, Forbidden, NotFound, Unavailable, Unprocessable
from ..interview_engine.lifecycle import find_owned
from ..api.deps import principal, unit, use

log = logging.getLogger("ii.voice")
router = APIRouter(prefix="/v1/voice")
MAX_AUDIO = 10 * 1024 * 1024
AUDIO_TYPES = {"audio/webm", "audio/ogg", "audio/mpeg", "audio/mp4", "audio/wav", "audio/x-wav", "audio/m4a",
               "audio/aac", "video/webm"}

# A spoken turn makes several speech calls (one per sentence). Re-checking access, the flag and
# the user row in the database on each one adds ~0.3 s per call across regions, so a positive
# check is remembered per user for a minute (revoking access takes effect within that minute;
# the interview routes themselves always check live).
GATE_TTL_S = 60.0
_gate_ok: Dict[str, float] = {}
_gate_lock = threading.Lock()


def reset_gate_cache() -> None:
    with _gate_lock:
        _gate_ok.clear()


flags.on_change(reset_gate_cache)  # switching voice off (or access changes) applies at once


def _require_voice(db) -> None:
    if not flags.flag(db, "voice.enabled"):
        raise Forbidden("Voice mode isn't available right now. You can type your answers.", code="voice_disabled")


def _voice_gate(p: Principal) -> None:
    key = str(p.user_id)
    now = time.monotonic()
    with _gate_lock:
        fresh = _gate_ok.get(key, 0.0) > now
    if fresh:
        rate_limit.check(key, "voice")
        return
    with unit() as db:
        use(db, p, klass="voice")
        _require_voice(db)
    with _gate_lock:
        _gate_ok[key] = now + GATE_TTL_S


@router.post("/transcribe")
def transcribe(audio: UploadFile = File(...), p: Principal = Depends(principal)):
    data = audio.file.read(MAX_AUDIO + 1)
    if len(data) > MAX_AUDIO:
        raise Unprocessable("That recording is too long. Keep answers under a few minutes.", code="audio_too_large")
    mime = (audio.content_type or "").split(";")[0].strip().lower()
    if mime not in AUDIO_TYPES:
        raise Unprocessable("Unsupported audio format.", code="bad_audio")
    _voice_gate(p)
    for provider, model in resolve("stt", "stt"):
        t0 = time.perf_counter()
        try:
            res = provider.transcribe(data, filename=audio.filename or "answer.webm", mime=mime, model=model,
                                      timeout_s=60.0)
        except ProviderError as e:
            record_media_run(stage="stt", provider=provider.name, model=model, user_id=p.user_id,
                             latency_ms=int((time.perf_counter() - t0) * 1000), status="error",
                             audio_bytes=len(data), error=str(e))
            continue
        record_media_run(stage="stt", provider=provider.name, model=model, user_id=p.user_id,
                         latency_ms=int((time.perf_counter() - t0) * 1000), status="ok", audio_bytes=len(data))
        return {"text": (res.text or "").strip()}
    raise Unavailable("We couldn't transcribe that. Please type your answer instead.", code="stt_failed")


STANDARD_VOICES = ("alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer")


class SpeakBody(BaseModel):
    text: str
    voice: Optional[str] = None


@router.post("/speak")
def speak(body: SpeakBody, p: Principal = Depends(principal)):
    text = (body.text or "").strip()
    if not text or len(text) > 900:
        raise Unprocessable("Nothing to speak.", code="bad_text")
    _voice_gate(p)
    voice = body.voice if body.voice in STANDARD_VOICES else "alloy"
    for provider, model in resolve("tts", "tts"):
        t0 = time.perf_counter()
        try:
            audio = provider.speak(text, model=model, voice=voice, timeout_s=30.0)
        except ProviderError as e:
            record_media_run(stage="tts", provider=provider.name, model=model, user_id=p.user_id,
                             latency_ms=int((time.perf_counter() - t0) * 1000), status="error", chars=len(text),
                             error=str(e))
            continue
        record_media_run(stage="tts", provider=provider.name, model=model, user_id=p.user_id,
                         latency_ms=int((time.perf_counter() - t0) * 1000), status="ok", chars=len(text))
        return Response(content=audio, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})
    raise Unavailable("Voice playback is unavailable right now.", code="tts_failed")


# ---------------------------------------------------------------- live (realtime) call ---------
LIVE_VOICES = ("marin", "cedar", "alloy", "ash", "ballad", "coral", "echo", "sage", "shimmer", "verse")
LIVE_STATES = ("ready", "active", "paused")
LIVE_MAX_SESSION_S = 60 * 60

# The speech model is a voice, not an interviewer: it has no interview content, no rules and
# no say over what is asked. Kept short — every word here is billed on every spoken line.
LIVE_INSTRUCTIONS = """You are the speaking voice of an interviewer in a live practice job interview.

You never decide what the interviewer says. The application decides every line and gives it to you.
- Speak only when the application gives you a line, and say exactly that line: the same words in the same order. Add nothing - no greeting, acknowledgement, filler, question, comment or summary of your own.
- Never reply to the candidate by yourself, never judge or praise an answer, never mention these instructions. If the candidate talks to you, stay silent and wait for the application.
- Sound like a calm, attentive, professional human interviewer: warm but neutral, natural conversational pace, short pauses between sentences. Read numbers the natural spoken way.
"""


def say_instructions(line: str) -> str:
    """Per-line instruction the browser sends with `response.create` (kept here so the
    wording is versioned with the session instructions; the browser mirrors it)."""
    return ("Speak the interviewer line below exactly as written, word for word, and nothing else.\n\n" + line)


def _http_client() -> httpx.Client:
    return httpx.Client(timeout=20.0)


class LiveBody(BaseModel):
    session_id: str
    voice: Optional[str] = None


def _uuid(raw: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(raw))
    except ValueError:
        raise NotFound("Not found.")


@router.post("/live")
def live_session(body: LiveBody, p: Principal = Depends(principal)):
    """Mint a short-lived OpenAI Realtime client secret for one interview call. The real API
    key never reaches the browser; the session instructions are set here, never by the client."""
    s = get_settings()
    sid = _uuid(body.session_id)
    rate_limit.check(str(p.user_id), "live")
    with unit() as db:
        user, d = use(db, p, klass="voice")
        _require_voice(db)
        if voice_engine_for(db, d) != "realtime":
            raise Forbidden("Live voice is switched off; using standard voice.", code="live_off")
        sess = find_owned(db, user.id, sid)
        if sess is None:
            raise NotFound("Interview not found.")
        if sess.status not in LIVE_STATES:
            raise Conflict("This interview isn't running.", code="not_live")
        budget = float(flags.flag(db, "limits.daily_budget_usd") or 0)
    if budget and spend_today_usd() >= budget:
        raise Unavailable("Live voice has reached today's capacity — using standard voice.", code="capacity")
    if not s.openai_api_key:
        raise Unavailable("Live voice isn't configured — using standard voice.", code="live_unconfigured")

    voice = body.voice if body.voice in LIVE_VOICES else "marin"
    eagerness = s.realtime_eagerness if s.realtime_eagerness in ("low", "medium", "high", "auto") else "low"
    payload = {"session": {
        "type": "realtime",
        "model": s.realtime_model,
        "instructions": LIVE_INSTRUCTIONS,
        "audio": {
            "input": {
                # Words-based end of turn: waits through a thinking pause, ends when the
                # sentence sounds finished. Auto-replies OFF (II decides every line).
                # Server-side interruption OFF: on laptop speakers the interviewer's own voice
                # leaks into the mic and would cut the interviewer off mid-sentence. The browser
                # cuts in instead, only when the words it hears are not the interviewer's own.
                "turn_detection": {"type": "semantic_vad", "eagerness": eagerness,
                                   "create_response": False, "interrupt_response": False},
                # English transcript (Indian English is otherwise often written in Devanagari).
                "transcription": {"model": s.realtime_transcribe_model, "language": "en"},
            },
            "output": {"voice": voice},
        },
    }}
    url = s.openai_base_url.rstrip("/") + "/realtime/client_secrets"
    headers = {"Authorization": f"Bearer {s.openai_api_key}", "Content-Type": "application/json"}
    t0 = time.perf_counter()
    try:
        with _http_client() as client:
            res = client.post(url, headers=headers, json=payload)
            if res.status_code in (400, 422) and s.realtime_transcribe_model != "whisper-1":
                log.warning("live: transcription model %s rejected; retrying with whisper-1", s.realtime_transcribe_model)
                payload["session"]["audio"]["input"]["transcription"] = {"model": "whisper-1", "language": "en"}
                res = client.post(url, headers=headers, json=payload)
    except httpx.HTTPError as e:
        record_media_run(stage="live_session", provider="openai", model=s.realtime_model, user_id=p.user_id,
                         latency_ms=int((time.perf_counter() - t0) * 1000), status="error", error=str(e))
        raise Unavailable("Couldn't start a live call — using standard voice.", code="live_failed")
    latency = int((time.perf_counter() - t0) * 1000)
    if res.status_code >= 400:
        log.warning("live: client_secrets %s: %s", res.status_code, res.text[:400])
        record_media_run(stage="live_session", provider="openai", model=s.realtime_model, user_id=p.user_id,
                         latency_ms=latency, status="error", error=f"{res.status_code}: {res.text[:300]}")
        raise Unavailable("Couldn't start a live call — using standard voice.", code="live_failed")
    data = res.json()
    cs = data.get("client_secret")  # beta shape: {"client_secret": {"value", "expires_at"}}
    secret = data.get("value") or (cs.get("value") if isinstance(cs, dict) else cs)
    expires_at = data.get("expires_at") or (cs.get("expires_at") if isinstance(cs, dict) else None)
    if not secret:
        raise Unavailable("Couldn't start a live call — using standard voice.", code="live_failed")
    record_media_run(stage="live_session", provider="openai", model=s.realtime_model, user_id=p.user_id,
                     latency_ms=latency, status="ok")
    return {"client_secret": secret, "expires_at": expires_at, "model": s.realtime_model,
            "voice": voice, "max_session_s": LIVE_MAX_SESSION_S,
            "say_prefix": say_instructions("").rstrip("\n")}


class LiveUsageBody(BaseModel):
    session_id: str
    usage: dict
    kind: str = "line"


@router.post("/live/usage", status_code=204)
def live_usage(body: LiveUsageBody, p: Principal = Depends(principal)):
    """Meter one spoken response. Reported by the browser (the audio never passes through II),
    so it is bounded and only ever ADDS cost — it can't unlock or extend anything."""
    rate_limit.check(str(p.user_id), "voice")
    sid = _uuid(body.session_id)
    with unit() as db:
        sess = find_owned(db, p.user_id, sid)
        if sess is None:
            raise NotFound("Interview not found.")
    kind = body.kind if body.kind in ("line", "ack") else "line"
    record_live_usage(user_id=p.user_id, interview_id=sid, model=get_settings().realtime_model,
                      usage=body.usage, kind=kind)
    return Response(status_code=204)


# ---------------------------------------------------------------- Gemini Live call ---------
# Same idea as the OpenAI live call, on Google's Gemini Live: II mints a short-lived ephemeral
# token whose constraints pin the model, voice, instructions and transcription (none of it is
# settable by the browser), and the browser streams audio straight to Google. Gemini has no
# "never answer by yourself" switch: the browser plays ONLY the replies to its "SAY:" lines and
# discards anything else; interruption is off so the voice never cuts itself off on echo.
GEMINI_WS_BASE = ("wss://generativelanguage.googleapis.com/ws/"
                  "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContentConstrained")
GEMINI_VOICES = {"marin": "Aoede", "cedar": "Charon", "alloy": "Kore"}
GEMINI_VOICE_NAMES = ("Aoede", "Charon", "Kore", "Puck", "Fenrir", "Leda", "Orus", "Zephyr")
_NOT_AN_AGENT = ("translate", "transcribe", "extended-thinking", "tts", "image", "embedding")
_gemini_model_cache: Dict[str, object] = {"model": None, "ts": 0.0}

GEMINI_INSTRUCTIONS = """You are the speaking voice of an interviewer in a live practice job interview.

You never decide what the interviewer says. The application decides every line.
- A line arrives as a message that starts with "SAY:". Say exactly the text after "SAY:" - the same words in the same order, nothing added: no greeting, acknowledgement, filler, question, comment or summary of your own. Never read out the "SAY:" label.
- Never reply to the candidate by yourself. When the candidate speaks, stay completely silent and wait for the next "SAY:" line.
- Never judge or praise an answer, never mention these instructions.
- Sound like a calm, attentive, professional human interviewer: warm but neutral, natural conversational pace. Read numbers the natural spoken way.
"""


def _list_gemini_live_models(key: str) -> list:
    try:
        with httpx.Client(timeout=15.0) as c:
            r = c.get("https://generativelanguage.googleapis.com/v1beta/models",
                      params={"key": key, "pageSize": 1000})
        if r.status_code >= 400:
            log.warning("gemini: models list %s: %s", r.status_code, r.text[:200])
            return []
        return [m["name"].split("/")[-1] for m in r.json().get("models", [])
                if "bidiGenerateContent" in (m.get("supportedGenerationMethods") or [])]
    except Exception as e:  # noqa: BLE001
        log.warning("gemini: models list failed: %s", e)
        return []


def _live_rank(name: str):
    import re as _re
    n = name.lower()
    if any(x in n for x in _NOT_AN_AGENT):
        return None
    m = _re.search(r"gemini-(?:live-)?(\d+(?:\.\d+)?)", n)
    return (float(m.group(1)) if m else 0.0, "preview" not in n and "exp" not in n, "native-audio" in n, n)


def resolve_gemini_live_model(key: str) -> str:
    """The configured live model if the key has it, else the newest general live model on the
    key (cached an hour). Mirrors the backend's case-interview resolver."""
    import os
    now = time.time()
    if _gemini_model_cache["model"] and now - float(_gemini_model_cache["ts"] or 0) < 3600:
        return str(_gemini_model_cache["model"])
    live = _list_gemini_live_models(key)
    env = (os.environ.get("II_GEMINI_LIVE_MODEL") or os.environ.get("GEMINI_LIVE_MODEL") or "").strip()
    chosen = env if env and (env in live or not live) else None
    if not chosen and live:
        ranked = [(r, n) for n in live if (r := _live_rank(n)) is not None]
        chosen = max(ranked)[1] if ranked else live[0]
    chosen = chosen or env or "gemini-3.8-live"
    _gemini_model_cache.update(model=chosen, ts=now)
    return chosen


def gemini_session_configs(model_id: str, voice: str) -> list:
    """Session configs, most tuned first; a config Google rejects falls through to a simpler one."""
    plain = {"response_modalities": ["AUDIO"], "system_instruction": GEMINI_INSTRUCTIONS,
             "input_audio_transcription": {}, "output_audio_transcription": {}}
    voiced = dict(plain, speech_config={"voice_config": {"prebuilt_voice_config": {"voice_name": voice}}})
    tuned = dict(voiced, realtime_input_config={"activity_handling": "NO_INTERRUPTION"})
    fast = dict(voiced)
    fast["input_audio_transcription"] = {"language_codes": ["en-IN"]}
    fast["realtime_input_config"] = {
        "activity_handling": "NO_INTERRUPTION",
        "automatic_activity_detection": {"start_of_speech_sensitivity": "START_SENSITIVITY_LOW",
                                         "end_of_speech_sensitivity": "END_SENSITIVITY_LOW",
                                         "silence_duration_ms": 1200},
    }
    if "native-audio" in model_id or "gemini-2.5" in model_id:
        fast["thinking_config"] = {"thinking_budget": 0}
    # Gemini Live bills the whole context on every turn. The voice needs no memory (every line
    # arrives in full), so a small sliding window keeps each turn's cost flat over a long call.
    windowed = dict(fast, context_window_compression={"trigger_tokens": 12000,
                                                      "sliding_window": {"target_tokens": 3000}})
    return [("windowed", windowed), ("fast", fast), ("tuned", tuned), ("plain", plain)]


def _mint_gemini_token(key: str, model_id: str, cfg: dict) -> str:
    """Ephemeral token with constraints, via the google-genai SDK (installed with the backend).
    Raw REST field names differ from the SDK's, so the SDK is the reliable path."""
    import datetime as _dt
    from google import genai  # lazy: keeps its import cost off every other route
    now = _dt.datetime.now(tz=_dt.timezone.utc)
    tok = genai.Client(api_key=key).auth_tokens.create(config={
        "uses": 2,  # one connect + one reconnect
        "expire_time": now + _dt.timedelta(minutes=30),
        "new_session_expire_time": now + _dt.timedelta(minutes=2),
        "live_connect_constraints": {"model": model_id, "config": cfg},
    })
    name = getattr(tok, "name", None)
    if not name:
        raise RuntimeError("token without a name")
    return name


class GeminiBody(BaseModel):
    session_id: str
    voice: Optional[str] = None
    tier: int = 0


@router.post("/gemini")
def gemini_session(body: GeminiBody, p: Principal = Depends(principal)):
    s = get_settings()
    sid = _uuid(body.session_id)
    rate_limit.check(str(p.user_id), "live")
    with unit() as db:
        user, d = use(db, p, klass="voice")
        _require_voice(db)
        if voice_engine_for(db, d) != "gemini":
            raise Forbidden("Gemini voice is switched off.", code="live_off")
        sess = find_owned(db, user.id, sid)
        if sess is None:
            raise NotFound("Interview not found.")
        if sess.status not in LIVE_STATES:
            raise Conflict("This interview isn't running.", code="not_live")
        budget = float(flags.flag(db, "limits.daily_budget_usd") or 0)
    if budget and spend_today_usd() >= budget:
        raise Unavailable("Live voice has reached today's capacity — using standard voice.", code="capacity")
    if not s.gemini_api_key:
        raise Unavailable("Gemini voice isn't configured — using standard voice.", code="live_unconfigured")
    voice = GEMINI_VOICES.get(body.voice or "") or (body.voice if body.voice in GEMINI_VOICE_NAMES else "Aoede")
    t0 = time.perf_counter()
    model_id = resolve_gemini_live_model(s.gemini_api_key)
    configs = gemini_session_configs(model_id, voice)
    start = max(0, min(int(body.tier or 0), len(configs) - 1))
    token, tier = None, start
    for i in range(start, len(configs)):
        label, cfg = configs[i]
        try:
            token, tier = _mint_gemini_token(s.gemini_api_key, model_id, cfg), i
            break
        except ImportError:
            raise Unavailable("Gemini voice isn't available on this server — using standard voice.",
                              code="gemini_unavailable")
        except Exception as e:  # noqa: BLE001 - try the next, simpler config
            log.warning("gemini: %s config rejected: %s", label, str(e)[:200])
    latency = int((time.perf_counter() - t0) * 1000)
    if not token:
        record_media_run(stage="live_session", provider="gemini", model=model_id, user_id=p.user_id,
                         latency_ms=latency, status="error", error="all session configs rejected")
        raise Unavailable("Couldn't start a Gemini call — using standard voice.", code="live_failed")
    record_media_run(stage="live_session", provider="gemini", model=model_id, user_id=p.user_id,
                     latency_ms=latency, status="ok")
    return {"token": token, "ws_url": f"{GEMINI_WS_BASE}?access_token={token}", "model": f"models/{model_id}",
            "voice": voice, "setup": {"model": f"models/{model_id}"}, "tier": tier, "tiers": len(configs),
            "max_session_s": LIVE_MAX_SESSION_S}


class GeminiUsageBody(BaseModel):
    session_id: str
    seconds_connected: float = 0
    seconds_spoken: float = 0


@router.post("/gemini/usage", status_code=204)
def gemini_usage(body: GeminiUsageBody, p: Principal = Depends(principal)):
    """Meter a stretch of a Gemini call (reported by the browser every minute and on close):
    seconds of candidate audio streamed to Google (only while they talk) and seconds of
    interviewer audio Google generated. Each report is clamped to 10 minutes."""
    rate_limit.check(str(p.user_id), "voice")
    sid = _uuid(body.session_id)
    with unit() as db:
        if find_owned(db, p.user_id, sid) is None:
            raise NotFound("Interview not found.")
    record_voice_minutes(user_id=p.user_id, interview_id=sid, provider="gemini",
                         model=str(_gemini_model_cache.get("model") or "gemini-live"),
                         minutes_in=max(0.0, min(float(body.seconds_connected or 0), 600.0)) / 60.0,
                         minutes_out=max(0.0, min(float(body.seconds_spoken or 0), 600.0)) / 60.0)
    return Response(status_code=204)
