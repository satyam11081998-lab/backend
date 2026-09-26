"""
Daily scheduler — generates exactly one Case and one Guesstimate per day.

Idempotent: safe to run multiple times per day. Only fills today's slot if empty.
Called by /cron/schedule-daily (GitHub Actions at 00:01 AM IST, or the admin panel).

2026-06-02 fix: the guesstimate is now a real `cases` row (type='guesstimate').
We store its id in daily_schedule.guesstimate_code (a free-text column — there is
no `guesstimates` table, so no FK/constraint exists on it). /daily/today resolves
that id back out of `cases`. No DB migration required.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional
from services.supabase_client import get_supabase_client
from services.content_generator import save_generated_content, GeneratorError

IST_OFFSET = timezone(timedelta(hours=5, minutes=30))


def today_in_ist() -> datetime:
    """Return today's date as a datetime in IST."""
    now_ist = datetime.now(IST_OFFSET)
    return now_ist.replace(hour=0, minute=0, second=0, microsecond=0)


def _newest_active(supabase, guesstimate: bool, market: str = "IN"):
    """Newest active case (or guesstimate) of ONE market's bank.

    The market filter matters from 0070 on: the US bank is seeded with fresh
    created_at timestamps, so an unfiltered "newest active case" would hand
    India's fallback daily a US case. Pre-0070 (no column) → the old
    unfiltered read, where every row is India anyway.
    """
    def _q(with_market: bool):
        q = supabase.table("cases").select("id").eq("is_active", True)
        q = q.eq("type", "guesstimate") if guesstimate else q.neq("type", "guesstimate")
        if with_market:
            q = q.eq("market", market)
        return q.order("created_at", desc=True).limit(1).execute()
    try:
        return _q(True)
    except Exception as exc:  # noqa: BLE001
        if "market" in str(exc).lower() and market == "IN":
            return _q(False)
        raise


def _fallback_from_existing(supabase) -> Optional[Dict[str, Any]]:
    """If AI generation fails, schedule EXISTING active cases so the daily
    surface is never empty (free-tier users can only attempt the daily pair).
    Returns None only if the bank has no case or no guesstimate at all."""
    try:
        c = _newest_active(supabase, guesstimate=False, market="IN")
        g = _newest_active(supabase, guesstimate=True, market="IN")
        case_id = (c.data or [None])[0]["id"] if c and c.data else None
        guess_id = (g.data or [None])[0]["id"] if g and g.data else None
        if case_id and guess_id:
            return {"case_id": case_id, "guesstimate_id": guess_id, "fallback": True}
    except Exception:
        pass
    return None


def fill_daily_schedule() -> Dict[str, Any]:
    """
    Generate and fill the daily_schedule table for TODAY only.

    - If a row exists for today, leave it (idempotent).
    - If no row exists, generate a new Case & Guesstimate via AI and insert them.
    """
    supabase = get_supabase_client()
    today_str = today_in_ist().date().isoformat()

    # Step 1: already scheduled?
    try:
        existing_res = (
            supabase.table("daily_schedule")
            .select("scheduled_date, case_id, guesstimate_code")
            .eq("scheduled_date", today_str)
            .execute()
        )
    except Exception as e:
        raise RuntimeError(f"Failed to fetch existing schedule: {e}")

    if existing_res and existing_res.data:
        return {
            "status": "ok",
            "message": f"Schedule already full for {today_str}",
            "filled": 0,
        }

    # Step 2: generate (case + guesstimate, both as cases rows)
    try:
        generated = save_generated_content()
    except Exception as e:
        # ANY generation failure (GeneratorError, OpenAI rate-limit/timeout, network,
        # JSON parse, etc.) must NOT leave the daily surface empty — free-tier users
        # can only attempt the daily pair. Fall back to existing active cases.
        generated = _fallback_from_existing(supabase)
        if not generated:
            raise RuntimeError(
                f"AI generation failed ({type(e).__name__}: {e}) and no fallback "
                f"case/guesstimate available in the bank"
            )

    # Step 3: insert today's schedule row.
    # guesstimate_code stores the guesstimate CASE id (resolved by /daily/today).
    try:
        supabase.table("daily_schedule").insert(
            {
                "scheduled_date": today_str,
                "case_id": generated["case_id"],
                "guesstimate_code": generated["guesstimate_id"],
                "brief_headline_id": None,  # brief tile uses the star headline, queried separately
            }
        ).execute()
    except Exception as e:
        raise RuntimeError(f"Failed to insert daily schedule: {e}")

    return {
        "status": "ok",
        "message": f"Generated and scheduled new case + guesstimate for {today_str}",
        "filled": 1,
        "details": generated,
    }



# =============================================================================
# International daily (US + Europe) — 2026-09-25
# =============================================================================
# Lives in market_daily_schedule (migration 0070), NOT daily_schedule, so the
# India table and its ~6 unfiltered readers are untouched. Keyed on the US
# Eastern calendar day. Same idempotency contract as fill_daily_schedule().

def _bank_pick_us(supabase) -> Optional[Dict[str, Any]]:
    """Fallback when generation fails: the curated US bank item scheduled
    least recently (never-scheduled first), one case + one guesstimate."""
    try:
        recent = (
            supabase.table("market_daily_schedule")
            .select("case_id, guesstimate_id")
            .eq("market", "US")
            .order("scheduled_date", desc=True)
            .limit(120)
            .execute()
        )
        used_order: Dict[str, int] = {}
        for i, r in enumerate(recent.data or []):
            for k in ("case_id", "guesstimate_id"):
                if r.get(k) and r[k] not in used_order:
                    used_order[r[k]] = i  # smaller = used more recently

        def pick(guesstimate: bool) -> Optional[str]:
            q = supabase.table("cases").select("id, created_at").eq("is_active", True).eq("market", "US")
            q = q.eq("type", "guesstimate") if guesstimate else q.neq("type", "guesstimate")
            rows = (q.order("created_at", desc=False).limit(500).execute().data) or []
            if not rows:
                return None
            never = [r["id"] for r in rows if r["id"] not in used_order]
            if never:
                return never[0]
            # all used: the one used longest ago (largest index)
            return max(rows, key=lambda r: used_order.get(r["id"], -1))["id"]

        cid, gid = pick(False), pick(True)
        if cid and gid:
            return {"case_id": cid, "guesstimate_id": gid, "fallback": True}
    except Exception:
        pass
    return None


def fill_market_daily_schedule(market: str = "US") -> Dict[str, Any]:
    """Generate (or, failing that, pick from the curated bank) today's
    international daily pair. Idempotent; a concurrent double-fire converges
    on whichever insert lands first (UNIQUE(market, scheduled_date))."""
    from services.markets import market_today  # local: keeps India import path unchanged

    if market != "US":
        raise ValueError(f"Unsupported market: {market}")
    supabase = get_supabase_client()
    today_str = market_today("US")

    existing = (
        supabase.table("market_daily_schedule")
        .select("scheduled_date")
        .eq("market", market)
        .eq("scheduled_date", today_str)
        .limit(1)
        .execute()
    )
    if existing and existing.data:
        return {"status": "ok", "message": f"{market} schedule already full for {today_str}", "filled": 0}

    source = "generated"
    try:
        generated = save_generated_content(market="US")
    except Exception as e:  # noqa: BLE001 — any generation failure → curated bank
        generated = _bank_pick_us(supabase)
        source = "bank"
        if not generated:
            raise RuntimeError(
                f"US generation failed ({type(e).__name__}: {e}) and the US bank is empty — "
                f"run supabase/seed-us-market.sql"
            )

    try:
        supabase.table("market_daily_schedule").insert(
            {
                "market": market,
                "scheduled_date": today_str,
                "case_id": generated["case_id"],
                "guesstimate_id": generated["guesstimate_id"],
                "source": source,
            }
        ).execute()
    except Exception as e:  # noqa: BLE001
        if "duplicate" in str(e).lower() or "23505" in str(e):
            return {"status": "ok", "message": f"{market} schedule filled concurrently for {today_str}", "filled": 0}
        raise RuntimeError(f"Failed to insert {market} daily schedule: {e}")

    return {
        "status": "ok",
        "message": f"Scheduled {market} case + guesstimate for {today_str} ({source})",
        "filled": 1,
        "details": {**generated, "source": source},
    }
