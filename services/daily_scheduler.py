"""
Daily scheduler — keeps a 7-day buffer of scheduled cases + guesstimates.

Idempotent: safe to run multiple times per day. Only fills empty slots.
Picks items not used in the last 60 days to ensure variety.

Called by /cron/schedule-daily endpoint at 6 AM IST daily.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Any
from services.supabase_client import get_supabase_client

# 7-day buffer — if cron fails for 6 consecutive days, day 7 still has content
BUFFER_DAYS = 7

# Don't reuse a case if it was scheduled in the last N days
REUSE_COOLDOWN_DAYS = 60

IST_OFFSET = timezone(timedelta(hours=5, minutes=30))


def today_in_ist() -> datetime:
    """Return today's date as a datetime in IST."""
    now_ist = datetime.now(IST_OFFSET)
    return now_ist.replace(hour=0, minute=0, second=0, microsecond=0)


def fill_daily_schedule() -> Dict[str, Any]:
    """
    Fill the daily_schedule table to maintain a BUFFER_DAYS-ahead inventory.
    
    For each of the next BUFFER_DAYS dates (starting today):
    - If a row exists for that date, leave it.
    - If no row exists, create one with a randomly-picked case + guesstimate
      that hasn't been scheduled in the last REUSE_COOLDOWN_DAYS.
    
    Returns a summary dict of what was created.
    """
    supabase = get_supabase_client()
    today = today_in_ist().date()
    
    # Step 1: Fetch existing schedule rows for the next BUFFER_DAYS
    end_date = today + timedelta(days=BUFFER_DAYS - 1)
    try:
        existing_res = supabase.table("daily_schedule") \
            .select("scheduled_date") \
            .gte("scheduled_date", today.isoformat()) \
            .lte("scheduled_date", end_date.isoformat()) \
            .execute()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch existing schedule: {e}")
    
    existing_dates = set()
    if existing_res and existing_res.data:
        for row in existing_res.data:
            existing_dates.add(row["scheduled_date"])
    
    # Step 2: Determine which dates need filling
    dates_to_fill = []
    for i in range(BUFFER_DAYS):
        d = (today + timedelta(days=i)).isoformat()
        if d not in existing_dates:
            dates_to_fill.append(d)
    
    if not dates_to_fill:
        return {
            "status": "ok",
            "message": "Schedule already full for next 7 days",
            "filled": 0,
            "buffer_days": BUFFER_DAYS,
        }
    
    # Step 3: Get pool of cases NOT scheduled in the cooldown window
    cooldown_start = today - timedelta(days=REUSE_COOLDOWN_DAYS)
    try:
        recent_case_res = supabase.table("daily_schedule") \
            .select("case_id") \
            .gte("scheduled_date", cooldown_start.isoformat()) \
            .execute()
        recently_used_case_ids = {
            row["case_id"] for row in (recent_case_res.data or [])
            if row.get("case_id")
        }
    except Exception as e:
        raise RuntimeError(f"Failed to fetch recent schedule: {e}")
    
    # Fetch all active cases
    try:
        cases_res = supabase.table("cases") \
            .select("id") \
            .eq("is_active", True) \
            .execute()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch cases: {e}")
    
    available_case_ids = [
        c["id"] for c in (cases_res.data or [])
        if c["id"] not in recently_used_case_ids
    ]
    
    # If pool is exhausted, allow reuse (better than crashing)
    if not available_case_ids:
        available_case_ids = [c["id"] for c in (cases_res.data or [])]
    
    if not available_case_ids:
        raise RuntimeError("No cases available — public.cases table empty?")
    
    # Step 4: Get pool of guesstimates similarly
    # Guesstimates come from lib/curriculum (frontend); for now we'll use a static list
    # that the scheduler picks from. This list lives in the frontend curriculum/data files.
    # 
    # We'll use a placeholder list of guesstimate codes. In a future iteration, we can
    # sync curriculum guesstimate codes into Supabase.
    
    # For Phase 2: leave guesstimate_code NULL for now. Frontend can fall back to
    # picking a curriculum guesstimate client-side if scheduled is NULL.
    # 
    # If you want to populate them properly, add a guesstimates table later.
    
    # Step 5: Fill empty dates
    import random
    filled = []
    for date_str in dates_to_fill:
        # Pick a random case
        case_id = random.choice(available_case_ids)
        # Remove from pool to avoid same-case-twice-in-a-row
        if len(available_case_ids) > 1:
            available_case_ids.remove(case_id)
        
        try:
            supabase.table("daily_schedule").insert({
                "scheduled_date": date_str,
                "case_id": case_id,
                "guesstimate_code": None,  # filled by frontend fallback for now
                "brief_headline_id": None,  # brief tile uses star headline of the day, queried separately
            }).execute()
            filled.append({"date": date_str, "case_id": case_id})
        except Exception as e:
            # Likely a unique-constraint violation if another cron ran simultaneously
            # Just skip — the slot is filled
            print(f"Skipping {date_str}: {e}")
            continue
    
    return {
        "status": "ok",
        "message": f"Filled {len(filled)} daily slots",
        "filled": len(filled),
        "buffer_days": BUFFER_DAYS,
        "details": filled,
    }
