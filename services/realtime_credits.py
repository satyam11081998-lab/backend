"""
Real-time interview minute CREDITS.

Real-time voice is ~10x pricier than the Groq pipeline, so it is metered by a
separate credit balance instead of the flat Pro sub:

  - Pro gets INCLUDED_MIN_PRO minutes per ~monthly period, refilled on read.
  - Beyond that, a user buys minute packs -> purchased_remaining (never expires).
  - Deduction burns INCLUDED first, then PURCHASED.

Every function fails SAFE: a credit-store hiccup must never crash a live
interview. get_balance returns zeros on error; deduct/add are best-effort.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional

from services.supabase_client import get_supabase_client

# Pro's monthly included real-time allowance (minutes). Env-tunable, no redeploy.
INCLUDED_MIN_PRO = float(os.getenv("REALTIME_INCLUDED_MIN_PRO", "60"))
PERIOD_DAYS = int(os.getenv("REALTIME_INCLUDED_PERIOD_DAYS", "30"))
# One-time real-time TRIAL for non-Pro users (minutes). Seeded on their first-ever
# balance read and NEVER refilled (only Pro's allowance refills), so it is a true
# lifetime grant — enough for ~2 seven-minute voice interviews. Env-tunable.
FREE_TRIAL_MIN = float(os.getenv("REALTIME_FREE_TRIAL_MIN", "14"))


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _row(supabase, user_id: str) -> Optional[dict]:
    try:
        res = supabase.table("realtime_credits").select("*").eq("user_id", user_id).maybe_single().execute()
        return res.data if (res and res.data) else None
    except Exception:
        return None


def _parse_ts(v) -> datetime:
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except Exception:
        return _now()


def get_balance(supabase, user_id: str, tier: str) -> Dict[str, Any]:
    """Balance for the user, refilling Pro's monthly included allowance if the
    period has elapsed (or the row is new). Fails safe -> zeros on any error."""
    is_pro = tier == "pro"
    refill_allowance = INCLUDED_MIN_PRO if is_pro else 0.0        # only Pro refills monthly
    first_grant = INCLUDED_MIN_PRO if is_pro else FREE_TRIAL_MIN  # one-time seed on a new row
    try:
        row = _row(supabase, user_id)
        now = _now()
        if row is None:
            included = first_grant    # Pro: full allowance; non-Pro: one-time free trial
            purchased = 0.0
            period_start = now
            _write(supabase, user_id, included, period_start, purchased)
        else:
            included = float(row.get("included_remaining") or 0)
            purchased = float(row.get("purchased_remaining") or 0)
            period_start = _parse_ts(row.get("included_period_start"))
            if is_pro and (now - period_start) >= timedelta(days=PERIOD_DAYS):
                included = refill_allowance  # new monthly cycle (Pro only; non-Pro never refills)
                period_start = now
                _write(supabase, user_id, included, period_start, purchased)
        return {
            "included_remaining": round(included, 2),
            "purchased_remaining": round(purchased, 2),
            "total_remaining": round(included + purchased, 2),
            "included_allowance": first_grant,
            "tier": tier,
        }
    except Exception:
        return {"included_remaining": 0, "purchased_remaining": 0, "total_remaining": 0,
                "included_allowance": (INCLUDED_MIN_PRO if is_pro else FREE_TRIAL_MIN), "tier": tier}


def has_credit(supabase, user_id: str, tier: str) -> bool:
    return get_balance(supabase, user_id, tier)["total_remaining"] > 0


def deduct(supabase, user_id: str, minutes: float) -> None:
    """Burn `minutes` from included first, then purchased. Best-effort."""
    if not minutes or minutes <= 0:
        return
    try:
        row = _row(supabase, user_id)
        if row is None:
            return
        inc = float(row.get("included_remaining") or 0)
        pur = float(row.get("purchased_remaining") or 0)
        m = float(minutes)
        take = min(inc, m); inc -= take; m -= take
        take = min(pur, m); pur -= take
        supabase.table("realtime_credits").update({
            "included_remaining": round(max(0.0, inc), 3),
            "purchased_remaining": round(max(0.0, pur), 3),
            "updated_at": _now().isoformat(),
        }).eq("user_id", user_id).execute()
    except Exception:
        return


def add_purchased(supabase, user_id: str, minutes: float) -> None:
    """Add purchased (top-up) minutes. Best-effort; creates the row if missing."""
    if not minutes or minutes <= 0:
        return
    try:
        row = _row(supabase, user_id)
        pur = float(row.get("purchased_remaining") or 0) if row else 0.0
        inc = float(row.get("included_remaining") or 0) if row else 0.0
        ps = row.get("included_period_start") if row else _now().isoformat()
        _write(supabase, user_id, inc, ps, pur + float(minutes))
    except Exception:
        return


def _write(supabase, user_id: str, included, period_start, purchased) -> None:
    ps = period_start.isoformat() if isinstance(period_start, datetime) else (period_start or _now().isoformat())
    supabase.table("realtime_credits").upsert({
        "user_id": user_id,
        "included_remaining": round(float(included), 3),
        "included_period_start": ps,
        "purchased_remaining": round(float(purchased), 3),
        "updated_at": _now().isoformat(),
    }, on_conflict="user_id").execute()
