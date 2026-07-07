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
import time
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
}
WHISPER_PER_MIN = 0.006

# ---------------------------------------------------------------------------
# Per-tier daily allowances (env-overridable — tune without a redeploy).
# Defaults chosen to keep worst-case per-user cost a small fraction of tier
# revenue (ROI-positive), while never blocking a normal practice day.
#   Voice: Whisper $0.006/min. Free 5 / Lite 20 / Pro 60 min-day.
#   OCR:   gpt-4o-mini vision. Free 5 / Lite 20 / Pro 100 images-day.
# ---------------------------------------------------------------------------
def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default

VOICE_MIN_PER_DAY = {
    "free": _int_env("AI_VOICE_MIN_FREE", 5),
    "lite": _int_env("AI_VOICE_MIN_LITE", 20),
    "pro":  _int_env("AI_VOICE_MIN_PRO", 60),
}
OCR_IMG_PER_DAY = {
    "free": _int_env("AI_OCR_IMG_FREE", 5),
    "lite": _int_env("AI_OCR_IMG_LITE", 20),
    "pro":  _int_env("AI_OCR_IMG_PRO", 100),
}

DAILY_BUDGET_USD = float(os.getenv("AI_DAILY_BUDGET_USD", "10.0"))

_IST = timezone(timedelta(hours=5, minutes=30))


def _ist_day_start_utc_iso() -> str:
    """UTC ISO timestamp of the most recent IST midnight (quota/budget window)."""
    now_ist = datetime.now(_IST)
    start_ist = now_ist.replace(hour=0, minute=0, second=0, microsecond=0)
    return start_ist.astimezone(timezone.utc).isoformat()


def _est_cost(model: str, pt: Optional[int], ct: Optional[int]) -> float:
    i, o = PRICES.get(model, (2.50, 10.00))
    return (pt or 0) * i / 1e6 + (ct or 0) * o / 1e6


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

        if model == "whisper-1" and audio_minutes is not None:
            cost = audio_minutes * WHISPER_PER_MIN
        else:
            cost = _est_cost(model, pt, ct)

        get_supabase_client().table("ai_usage_log").insert({
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
        }).execute()
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


def get_ai_input_quota(supabase, user_id: str) -> Dict[str, Any]:
    """Full quota snapshot for the frontend 'X min / Y images left today' UI."""
    tier = effective_tier(supabase, user_id)
    v_limit = VOICE_MIN_PER_DAY.get(tier, VOICE_MIN_PER_DAY["free"])
    o_limit = OCR_IMG_PER_DAY.get(tier, OCR_IMG_PER_DAY["free"])
    v_used = voice_minutes_used_today(supabase, user_id)
    o_used = ocr_images_used_today(supabase, user_id)
    return {
        "tier": tier,
        "voice": {
            "used_min": round(v_used, 2),
            "limit_min": v_limit,
            "remaining_min": max(0.0, round(v_limit - v_used, 2)),
        },
        "images": {
            "used": o_used,
            "limit": o_limit,
            "remaining": max(0, o_limit - o_used),
        },
    }


def assert_voice_quota(supabase, user_id: str) -> float:
    """Raise 429 if today's voice minutes are used up. Returns remaining minutes."""
    q = get_ai_input_quota(supabase, user_id)["voice"]
    if q["remaining_min"] <= 0:
        raise HTTPException(
            status_code=429,
            detail=f"Daily voice-to-text limit reached ({q['limit_min']} min). "
                   f"Resets at midnight IST — you can still type your answer.",
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
def spend_today_usd() -> float:
    try:
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
    except Exception:
        return 0.0


def assert_daily_budget() -> None:
    """Raise 503 once estimated spend for the IST day crosses the budget.
    Backstop only — per-user quotas + auth are the primary defenses. Fail-open
    if the check itself errors (never take the product down on a budget-read bug)."""
    if DAILY_BUDGET_USD <= 0:
        return
    try:
        if spend_today_usd() >= DAILY_BUDGET_USD:
            raise HTTPException(
                status_code=503,
                detail="AI features are paused for today (daily spend cap reached). "
                       "They'll resume automatically after midnight IST.",
            )
    except HTTPException:
        raise
    except Exception:
        return
