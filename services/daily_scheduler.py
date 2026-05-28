"""
Daily scheduler — generates exactly one Case and Guesstimate per day at 00:01 AM.

Idempotent: safe to run multiple times per day. Only fills today's slot if empty.
Called by /cron/schedule-daily endpoint at 00:01 AM IST daily.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Any
from services.supabase_client import get_supabase_client
from services.content_generator import save_generated_content, GeneratorError

IST_OFFSET = timezone(timedelta(hours=5, minutes=30))

def today_in_ist() -> datetime:
    """Return today's date as a datetime in IST."""
    now_ist = datetime.now(IST_OFFSET)
    return now_ist.replace(hour=0, minute=0, second=0, microsecond=0)

def fill_daily_schedule() -> Dict[str, Any]:
    """
    Generate and fill the daily_schedule table for TODAY only.
    
    - If a row exists for today, leave it.
    - If no row exists, generate a new Case & Guesstimate via AI and insert them.
    """
    supabase = get_supabase_client()
    today_str = today_in_ist().date().isoformat()
    
    # Step 1: Check if today is already scheduled
    try:
        existing_res = supabase.table("daily_schedule") \
            .select("scheduled_date, case_id, guesstimate_code") \
            .eq("scheduled_date", today_str) \
            .execute()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch existing schedule: {e}")
        
    if existing_res and existing_res.data:
        return {
            "status": "ok",
            "message": f"Schedule already full for {today_str}",
            "filled": 0,
        }
        
    # Step 2: Generate new Case and Guesstimate using AI
    try:
        generated = save_generated_content()
    except GeneratorError as e:
        raise RuntimeError(f"AI Generation failed: {e}")
        
    # Step 3: Insert into daily_schedule for today
    try:
        supabase.table("daily_schedule").insert({
            "scheduled_date": today_str,
            "case_id": generated["case_id"],
            "guesstimate_code": generated["guesstimate_code"],
            "brief_headline_id": None,  # brief tile uses star headline of the day, queried separately
        }).execute()
    except Exception as e:
        raise RuntimeError(f"Failed to insert daily schedule: {e}")
        
    return {
        "status": "ok",
        "message": f"Generated and scheduled new case/guesstimate for {today_str}",
        "filled": 1,
        "details": generated,
    }
