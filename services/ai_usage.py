"""
AI usage ledger + spend controls.

Three jobs, all fail-SAFE for the product (a logging/quota bug must never break a
legitimate request), but fail-CLOSED for spend (when a hard limit is truly reached,
we stop the call):

  1. log_ai_usage(...)         — write one row per billed OpenAI call to ai_usage_log.
  2. per-user daily quotas     — voice minutes (Whisper) and OCR images, by tier.
  3. assert_daily_budget()     — global kill switch: pause AI once the estimated
                                 day's spend crosses AI_DAILY_BUDGET_USD.

All prices are $/1M tokens (chat) or $/min (whisper). Update PRICES if OpenAI's
list changes; costs here are ESTIMATES for guardrails, not billing truth.
"""

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

from fastapi import HTTPException

from services.supabase_client import get_supabase_client
from services.access_guard import effective_tier

# ---------------------------------------------------------------------------
# Pricing (guardrail estimates)
# ---------------------------------------------------------------------------
PRICES = {  # (input $/1M, output $/1M)
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    # Groq Llama 3.3 70B — the interviewer LLM when GROQ_API_KEY is set. Cheap and
    # fast; kept here so its turns aren't mispriced against the gpt-4o default.
    "llama-3.3-70b-versatile": (0.59, 0.79),
}
# Gemini text models. WITHOUT an entry here _est_cost() falls through to its
# gpt-4o default of (2.50, 10.00) and books roughly $0.02 for a call that cost
# nothing — and since spend_today_usd() feeds assert_daily_budget(), a day of
# free-tier briefs would cross AI_DAILY_BUDGET_USD and 503 *every* AI feature in
# the product. Priced at zero for the free tier; override via env if you move to
# a paid Gemini tier so the kill switch keeps seeing real money.
GEMINI_TEXT_IN_PER_1M = float(os.getenv("GEMINI_TEXT_IN_PER_1M", "0.0"))
GEMINI_TEXT_OUT_PER_1M = float(os.getenv("GEMINI_TEXT_OUT_PER_1M", "0.0"))

WHISPER_PER_MIN = 0.006
# Groq's whisper-large-v3-turbo is ~$0.04/hr ≈ $0.000667/min — ~9x cheaper than
# OpenAI Whisper, same underlying model. Used when transcribe routes to Groq.
GROQ_WHISPER_PER_MIN = float(os.getenv("GROQ_WHISPER_PER_MIN", "0.000667"))
# OpenAI realtime transcription (gpt-live-transcribe), billed per minute of streamed
# audio ($0.017/min at 2026-09-29, developers.openai.com/api/docs/models/gpt-live-transcribe).
# Its rows are written under the SAME '/transcribe' endpoint as Whisper, so the
# existing per-user voice-minute quota keeps applying unchanged.
LIVE_TRANSCRIBE_PER_MIN = float(os.getenv("LIVE_TRANSCRIBE_PER_MIN", "0.017"))

# TTS is priced per CHARACTER, not per token — it must NOT go in PRICES (those
# are token tuples and _est_cost would read a TTS call as 0, because a speech
# response carries no `usage`). A silently-zero cost would make voice mode
# invisible to spend_today_usd(), which is exactly what assert_daily_budget()
# reads — the global kill switch would not see the most expensive thing we do.
# tts-1 is $15/1M chars; at ~900 spoken chars/min that is ~$0.0135/min.
TTS_CHARS_PER_MIN = 900.0
TTS_PER_MIN = 0.015
# Google WaveNet TTS: $4 / 1M chars (early-2026 price) ≈ $0.0036/min at 900
# chars/min — ~3.75x cheaper than OpenAI tts-1. Used when /speak routes to Google;
# without its own branch a WaveNet row would book $0 (no usage object, name isn't
# "tts...") and hide talk-mode spend from the daily-budget kill switch.
GOOGLE_TTS_PER_MIN = float(os.getenv("GOOGLE_TTS_PER_MIN", "0.0036"))
# Gemini Live speech-to-speech ~ $0.02-0.05/min all-in; used for /realtime-gemini rows.
GEMINI_LIVE_PER_MIN = float(os.getenv("GEMINI_LIVE_PER_MIN", "0.04"))

# Realtime (speech-to-speech) audio pricing, verified 2026-08-16.
# Input $32/1M audio tokens, output $64/1M. Rates: user audio is 1 token per
# 100ms (600 tok/min), assistant audio 1 token per 50ms (1200 tok/min).
# Priced per TOKEN, like TTS is priced per character — neither belongs in
# PRICES, which _est_cost() reads as (input, output) per-1M-TEXT-token tuples.
REALTIME_AUDIO_IN_PER_1M = 32.00
REALTIME_AUDIO_OUT_PER_1M = 64.00
REALTIME_IN_TOK_PER_MIN = 600.0
REALTIME_OUT_TOK_PER_MIN = 1200.0

# ---------------------------------------------------------------------------
# Per-tier daily allowances (env-overridable — tune without a redeploy).
# Defaults chosen to keep worst-case per-user cost a small fraction of tier
# revenue (ROI-positive), while never blocking a normal practice day.
#   Voice: Whisper $0.006/min. Free 5 / Lite 20 / Pro 60 min-day.
#   OCR:   gpt-4o-mini vision. Free 5 / Lite 20 / Pro 100 images-day.
# ---------------------------------------------------------------------------
def _nonneg_env(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default

VOICE_MIN_PER_DAY = {
    "free": _int_env("AI_VOICE_MIN_FREE", 5),
    "lite": _int_env("AI_VOICE_MIN_LITE", 20),
    "pro":  _int_env("AI_VOICE_MIN_PRO", 180),
}
OCR_IMG_PER_DAY = {
    "free": _int_env("AI_OCR_IMG_FREE", 5),
    "lite": _int_env("AI_OCR_IMG_LITE", 20),
    "pro":  _int_env("AI_OCR_IMG_PRO", 100),
}
# Talk mode (Pro only). This meter is SEPARATE from VOICE_MIN_PER_DAY: that one
# counts the candidate's speech into Whisper, this one counts the interviewer's
# speech out of TTS. Keeping them apart means dictation and talk mode do not
# quietly eat each other's allowance.
#
# NOTE: a full spoken case is 20-40 min of candidate audio, so talk mode also
# consumes VOICE_MIN_PER_DAY. Raise AI_VOICE_MIN_PRO alongside AI_TTS_MIN_PRO
# (both env-tunable) before unflagging talk mode, or a Pro user's second case of
# the day 429s mid-interview.
TTS_MIN_PER_DAY = {
    "free": _int_env("AI_TTS_MIN_FREE", 0),
    "lite": _int_env("AI_TTS_MIN_LITE", 0),
    "pro":  _int_env("AI_TTS_MIN_PRO", 180),
}

DAILY_BUDGET_USD = float(os.getenv("AI_DAILY_BUDGET_USD", "10.0"))

_IST = timezone(timedelta(hours=5, minutes=30))


def _ist_day_start_utc_iso() -> str:
    """UTC ISO timestamp of the most recent IST midnight (quota/budget window)."""
    now_ist = datetime.now(_IST)
    start_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_ist.astimezone(timezone.utc).isoformat()


def _est_cost(model: str, pt: Optional[int], ct: Optional[int]) -> float:
    name = (model or "").lower()
    # Gemini text models are matched by PREFIX, not by exact name: Google retires
    # and renames them (2.0-flash -> 3.x-flash) and an unrecognised name would
    # otherwise be billed at the gpt-4o fallback rate. "live" is excluded — those
    # rows are audio-priced in log_ai_usage() above and never reach here.
    if name.startswith("gemini") and "live" not in name:
        i, o = GEMINI_TEXT_IN_PER_1M, GEMINI_TEXT_OUT_PER_1M
    else:
        i, o = PRICES.get(model, (2.50, 10.00))
    return (pt or 0) * i / 1e6 + (ct or 0) * o / 1e6


# ---------------------------------------------------------------------------
# Off-the-hot-path ledger writes (SPEED, 2026-09-28)
# ---------------------------------------------------------------------------
# Every interviewer turn, TTS sentence and transcript logs a row here, and the
# insert used to run INLINE: an Oregon->Tokyo round trip (plus, before the shared
# client, a fresh TLS handshake) added to the reply the candidate is waiting on.
# The row is now built inline (cheap, so nothing about its content changes) and
# written by a small background pool. Still best-effort, exactly as before: a
# failed write is dropped silently. AI_USAGE_LOG_SYNC=1 restores inline writes.
_LOG_SYNC = os.getenv("AI_USAGE_LOG_SYNC", "").strip().lower() in ("1", "true", "yes", "on")
_LOG_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ai-usage-log")


def _insert_usage_row(row: dict) -> None:
    try:
        get_supabase_client().table("ai_usage_log").insert(row).execute()
    except Exception:
        return  # logging must never break the product


def _write_usage_row(row: dict) -> None:
    _note_spend(row.get("est_cost_usd") or 0.0)
    if _LOG_SYNC:
        _insert_usage_row(row)
        return
    try:
        _LOG_POOL.submit(_insert_usage_row, row)
    except Exception:  # pool shut down (process exiting) -> write inline
        _insert_usage_row(row)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def log_ai_usage(
    user_id: Optional[str] = None,
    endpoint: str = "",
    model: str = "",
    response: Any = None,
    audio_minutes: Optional[float] = None,
    latency_ms: Optional[int] = None,
    success: bool = True,
    meta: Optional[dict] = None,
) -> None:
    """Best-effort insert into ai_usage_log. NEVER raises."""
    try:
        pt = ct = tt = None
        openai_id = None
        if response is not None:
            usage = getattr(response, "usage", None)
            pt = getattr(usage, "prompt_tokens", None)
            ct = getattr(usage, "completion_tokens", None)
            tt = getattr(usage, "total_tokens", None)
            openai_id = getattr(response, "id", None)

        if "live-transcribe" in model and audio_minutes is not None:
            cost = audio_minutes * LIVE_TRANSCRIBE_PER_MIN
        elif "whisper" in model and audio_minutes is not None:
            # OpenAI Whisper is $0.006/min; Groq's whisper-large-v3* is ~9x cheaper.
            # Any non-"whisper-1" whisper model is assumed to be the Groq endpoint.
            cost = audio_minutes * (WHISPER_PER_MIN if model == "whisper-1" else GROQ_WHISPER_PER_MIN)
        elif "live" in model.lower() and audio_minutes is not None:
            # Gemini Live (gemini-*-live-*) — priced per minute of audio.
            cost = audio_minutes * GEMINI_LIVE_PER_MIN
        elif "wavenet" in model.lower() and audio_minutes is not None:
            # Google WaveNet /speak row — character-priced, no usage object, and
            # its model name ("en-US-Wavenet-D") is neither "tts..." nor "whisper".
            cost = audio_minutes * GOOGLE_TTS_PER_MIN
        elif model.startswith("tts") and audio_minutes is not None:
            # Character-priced, no usage object — see TTS_PER_MIN. Without this
            # branch every /speak row would book $0 and hide talk-mode spend
            # from the daily-budget kill switch.
            cost = audio_minutes * TTS_PER_MIN
        else:
            cost = _est_cost(model, pt, ct)

        _write_usage_row({
            "user_id": user_id,
            "endpoint": endpoint,
            "model": model,
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "total_tokens": tt,
            "audio_minutes": round(audio_minutes, 3) if audio_minutes is not None else None,
            "est_cost_usd": round(cost, 6),
            "latency_ms": latency_ms,
            "success": success,
            "openai_id": openai_id,
            "meta": meta or {},
        })
    except Exception:
        return  # logging must never break the product


# ---------------------------------------------------------------------------
# Per-user daily quotas (voice minutes + OCR images)
# ---------------------------------------------------------------------------
def _rows_today(supabase, user_id: str, endpoint: str):
    try:
        res = (
            supabase.table("ai_usage_log")
            .select("audio_minutes")
            .eq("user_id", user_id)
            .eq("endpoint", endpoint)
            .gte("created_at", _ist_day_start_utc_iso())
            .execute()
        )
        return res.data or []
    except Exception:
        return []


def voice_minutes_used_today(supabase, user_id: str) -> float:
    return round(sum(float(r.get("audio_minutes") or 0) for r in _rows_today(supabase, user_id, "/transcribe")), 3)


def ocr_images_used_today(supabase, user_id: str) -> int:
    return len(_rows_today(supabase, user_id, "/extract-text"))


def log_realtime_usage(
    user_id: Optional[str],
    input_tokens: int,
    output_tokens: int,
    meta: Optional[dict] = None,
) -> None:
    """Book the cost of one realtime turn so the daily-budget guard can see it.

    Realtime usage arrives from the BROWSER (the `response.done` event), which
    is the one place in the product where spend is reported by the client. That
    makes this function load-bearing: if it books zero — the way a TTS call
    would have, had it been routed through _est_cost() — then
    `spend_today_usd()` under-reports and `assert_daily_budget()` never trips.
    """
    cost = (
        input_tokens * REALTIME_AUDIO_IN_PER_1M / 1e6
        + output_tokens * REALTIME_AUDIO_OUT_PER_1M / 1e6
    )
    # Recorded as minutes too, so the /realtime meter reads like the others.
    minutes = (
        input_tokens / REALTIME_IN_TOK_PER_MIN + output_tokens / REALTIME_OUT_TOK_PER_MIN
    )
    try:
        _write_usage_row({
            "user_id": user_id,
            "endpoint": "/realtime",
            "model": os.getenv("REALTIME_MODEL", "gpt-realtime-2.1"),
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "audio_minutes": round(minutes, 3),
            "est_cost_usd": round(cost, 6),
            "success": True,
            "meta": meta or {},
        })
    except Exception:
        pass  # never let metering break a live interview


def realtime_minutes_used_today(supabase, user_id: str) -> float:
    """Realtime voice minutes today. Unmetered per-user by owner decision
    (2026-08-16) — exposed so a cap can be switched on from data, not guesswork."""
    return round(sum(float(r.get("audio_minutes") or 0) for r in _rows_today(supabase, user_id, "/realtime")), 3)


def speak_minutes_used_today(supabase, user_id: str) -> float:
    """Interviewer TTS minutes spoken to this user today (talk mode).

    Reads `/speak` rows only, so it cannot pollute — or be polluted by — the
    `/transcribe` voice meter above.
    """
    return round(sum(float(r.get("audio_minutes") or 0) for r in _rows_today(supabase, user_id, "/speak")), 3)


# Pool tasks here only ever READ; none waits on another pool task, so the pool
# can never deadlock itself however busy it gets.
_QUOTA_POOL = ThreadPoolExecutor(max_workers=12, thread_name_prefix="quota-read")


def get_ai_input_quota(supabase, user_id: str) -> Dict[str, Any]:
    """Full quota snapshot for the frontend 'X min / Y images left today' UI."""
    # SPEED: the tier read and the three meters are independent. Start the
    # meters (each already fails soft to 0) and read the tier meanwhile: one
    # round trip instead of four, same values.
    fv = _QUOTA_POOL.submit(voice_minutes_used_today, supabase, user_id)
    fo = _QUOTA_POOL.submit(ocr_images_used_today, supabase, user_id)
    fs = _QUOTA_POOL.submit(speak_minutes_used_today, supabase, user_id)
    tier = effective_tier(supabase, user_id)
    # Pipeline talk mode (Groq STT + Groq LLM + WaveNet TTS) is CHEAP (~Rs 0.4/min),
    # so it is included and unlimited for Pro. Real-time (Gemini/OpenAI) is the
    # expensive one and is metered separately by the realtime CREDIT system, not here.
    pipeline_unlimited = (tier == "pro")
    v_limit = VOICE_MIN_PER_DAY.get(tier, VOICE_MIN_PER_DAY["free"])
    o_limit = OCR_IMG_PER_DAY.get(tier, OCR_IMG_PER_DAY["free"])
    s_limit = TTS_MIN_PER_DAY.get(tier, TTS_MIN_PER_DAY["free"])
    v_used, o_used, s_used = fv.result(), fo.result(), fs.result()
    return {
        "tier": tier,
        "voice": {
            "used_min": round(v_used, 2),
            "limit_min": v_limit,
            "remaining_min": max(0.0, round(v_limit - v_used, 2)),
            "unlimited": pipeline_unlimited,
        },
        "images": {
            "used": o_used,
            "limit": o_limit,
            "remaining": max(0, o_limit - o_used),
        },
        # Additive key (talk mode). Existing "voice" and "images" blocks are
        # untouched, so every current reader keeps working unchanged.
        "speak": {
            "used_min": round(s_used, 2),
            "limit_min": s_limit,
            "remaining_min": max(0.0, round(s_limit - s_used, 2)),
            "unlimited": pipeline_unlimited,
        },
    }


# Per-sentence TTS gate (SPEED, 2026-09-28). /speak runs once per SENTENCE the
# interviewer says and only needs "is this user Pro, and is their talk-mode meter
# open" -- which for Pro is always unlimited. Re-reading the full snapshot every
# sentence put a Tokyo round trip in front of each sentence's audio. A snapshot
# is reused for QUOTA_CACHE_SECONDS (default 20; 0 disables); a plan change
# reaches /speak within that window.
_QUOTA_TTL = _nonneg_env("QUOTA_CACHE_SECONDS", 20.0)
_quota_cache: Dict[str, Any] = {}
_quota_lock = threading.Lock()


def get_ai_input_quota_cached(supabase, user_id: str) -> Dict[str, Any]:
    if _QUOTA_TTL <= 0:
        return get_ai_input_quota(supabase, user_id)
    now = time.monotonic()
    with _quota_lock:
        hit = _quota_cache.get(user_id)
    if hit is not None and now - hit[0] < _QUOTA_TTL:
        return hit[1]
    snap = get_ai_input_quota(supabase, user_id)
    with _quota_lock:
        if len(_quota_cache) > 5000:
            _quota_cache.clear()
        _quota_cache[user_id] = (now, snap)
    return snap


def check_voice_quota(snapshot: Dict[str, Any]) -> float:
    """assert_voice_quota() on a snapshot the caller already holds (no DB read)."""
    q = snapshot["voice"]
    if not q.get("unlimited") and q["remaining_min"] <= 0:
        raise HTTPException(
            status_code=429,
            detail=f"Daily voice-to-text limit reached ({q['limit_min']} min). "
                   f"Resets at midnight IST — you can still type your answer.",
        )
    return q["remaining_min"]


def assert_voice_quota(supabase, user_id: str) -> float:
    """Raise 429 if today's voice minutes are used up. Returns remaining minutes."""
    return check_voice_quota(get_ai_input_quota(supabase, user_id))


def quota_after_voice(snapshot: Dict[str, Any], minutes: float) -> Dict[str, Any]:
    """The snapshot as it reads once `minutes` more of transcription are booked --
    what a fresh get_ai_input_quota() would return after this call's log row,
    computed locally instead of four more database reads."""
    out = dict(snapshot)
    v = dict(snapshot["voice"])
    used = round(float(v.get("used_min") or 0) + float(minutes or 0), 2)
    v["used_min"] = used
    v["remaining_min"] = max(0.0, round(float(v["limit_min"]) - used, 2))
    out["voice"] = v
    return out


def assert_tts_quota(supabase, user_id: str) -> float:
    """Raise 429 if today's talk-mode minutes are used up. Returns remaining minutes.

    Kept for symmetry with assert_voice_quota / assert_ocr_quota and for any
    caller that does not already hold a quota snapshot. `/speak` deliberately
    does NOT use it: that route needs the tier AND the meter AND a remaining
    figure for its response header, and calling three separate helpers costs
    nine Supabase round-trips per spoken sentence. It takes one snapshot and
    reads all three off it.
    """
    q = get_ai_input_quota(supabase, user_id)["speak"]
    if not q.get("unlimited") and q["remaining_min"] <= 0:
        raise HTTPException(
            status_code=429,
            detail=f"Daily voice-interview limit reached ({q['limit_min']} min). "
                   f"Resets at midnight IST — you can carry on in the chat.",
        )
    return q["remaining_min"]


def assert_ocr_quota(supabase, user_id: str) -> int:
    """Raise 429 if today's OCR images are used up. Returns remaining images."""
    q = get_ai_input_quota(supabase, user_id)["images"]
    if q["remaining"] <= 0:
        raise HTTPException(
            status_code=429,
            detail=f"Daily image-scan limit reached ({q['limit']} images). "
                   f"Resets at midnight IST — you can still type your answer.",
        )
    return q["remaining"]


# ---------------------------------------------------------------------------
# Global daily-budget kill switch (catastrophe backstop)
# ---------------------------------------------------------------------------
def _read_spend_today() -> float:
    """One fresh read of today's estimated spend. RAISES on a failed read, so the
    cache below can tell 'spent nothing' from 'could not read'."""
    rows = (
        get_supabase_client()
        .table("ai_usage_log")
        .select("est_cost_usd")
        .gte("created_at", _ist_day_start_utc_iso())
        .execute()
        .data
        or []
    )
    return round(sum(float(r.get("est_cost_usd") or 0) for r in rows), 4)


def spend_today_usd() -> float:
    try:
        return _read_spend_today()
    except Exception:
        return 0.0


# --- Cached spend for the per-turn budget check (SPEED, 2026-09-28) -----------
# assert_daily_budget() runs before EVERY interviewer turn, TTS sentence and
# transcription, and it summed every ai_usage_log row of the day -- a full scan
# across the Pacific on the hot path, growing all day. It is a catastrophe
# backstop, not a meter, so it now reads a cached figure:
#   * fresh read at most every AI_BUDGET_CACHE_SECONDS (default 30), refreshed
#     in the BACKGROUND while the last figure is served (stale-while-revalidate);
#   * every cost this process logs is added to the cached figure immediately,
#     so our own spend is never stale;
#   * the figure resets at IST midnight, and one older than 5 minutes is never
#     served -- the request re-reads inline instead.
# AI_BUDGET_CACHE_SECONDS=0 restores a fresh read on every check.
_SPEND_TTL = _nonneg_env("AI_BUDGET_CACHE_SECONDS", 30.0)
_SPEND_MAX_STALE = 300.0
_spend_lock = threading.Lock()
_spend = {"day": None, "value": None, "at": 0.0, "refreshing": False}


def _note_spend(cost) -> None:
    try:
        c = float(cost or 0.0)
    except (TypeError, ValueError):
        return
    if c <= 0:
        return
    with _spend_lock:
        if _spend["value"] is not None and _spend["day"] == _ist_day_start_utc_iso():
            _spend["value"] = round(_spend["value"] + c, 6)


def _store_spend(day: str, value: float) -> None:
    with _spend_lock:
        _spend.update(day=day, value=value, at=time.monotonic())


def _refresh_spend_in_background(day: str) -> None:
    def job():
        try:
            _store_spend(day, _read_spend_today())
        except Exception:
            pass  # keep serving the last figure; the next check tries again
        finally:
            with _spend_lock:
                _spend["refreshing"] = False
    try:
        _LOG_POOL.submit(job)
    except Exception:
        with _spend_lock:
            _spend["refreshing"] = False


def _spend_for_budget_check() -> float:
    if _SPEND_TTL <= 0:
        return spend_today_usd()
    day = _ist_day_start_utc_iso()
    now = time.monotonic()
    with _spend_lock:
        value, same_day, age = _spend["value"], _spend["day"] == day, now - _spend["at"]
        if value is not None and same_day and age < _SPEND_MAX_STALE:
            if age >= _SPEND_TTL and not _spend["refreshing"]:
                _spend["refreshing"] = True
                start_refresh = True
            else:
                start_refresh = False
        else:
            start_refresh = None  # nothing usable cached: read inline
    if start_refresh is None:
        try:
            fresh = _read_spend_today()
        except Exception:
            return 0.0  # same fail-open as spend_today_usd()
        _store_spend(day, fresh)
        return fresh
    if start_refresh:
        _refresh_spend_in_background(day)
    return value


def assert_daily_budget() -> None:
    """Raise 503 once estimated spend for the IST day crosses the budget.
    Backstop only — per-user quotas + auth are the primary defenses. Fail-open
    if the check itself errors (never take the product down on a budget-read bug)."""
    if DAILY_BUDGET_USD <= 0:
        return
    try:
        if _spend_for_budget_check() >= DAILY_BUDGET_USD:
            raise HTTPException(
                status_code=503,
                detail="AI features are paused for today (daily spend cap reached). "
                       "They'll resume automatically after midnight IST.",
            )
    except HTTPException:
        raise
    except Exception:
        return
