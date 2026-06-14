"""
Cron-triggered routes for the GD briefs system.

Three endpoints called by the scheduler (GitHub Actions, daily 6 AM IST):
1. POST /cron/fetch-news     — daily: fetch + classify headlines
2. POST /cron/cleanup        — daily: delete headlines older than 14 days
3. POST /cron/schedule-daily — daily: fill today's case + guesstimate schedule

All require the X-Cron-Secret header to match CRON_SECRET env var.
This prevents random users from triggering expensive operations.

The actual fetch/classify/store logic lives in services/news_pipeline.py so that
the scheduled job here and the /news/headlines self-heal share one implementation.
"""

import os
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
from services.supabase_client import get_supabase_client
from services.daily_scheduler import fill_daily_schedule
from services.news_pipeline import run_news_refresh


router = APIRouter(prefix="/cron", tags=["cron"])


class CronResponse(BaseModel):
    """Generic response for cron endpoints."""
    status: str
    message: str
    details: dict = {}


def verify_cron_secret(x_cron_secret: Optional[str]) -> None:
    """Raise 401 if the request doesn't have a valid cron secret."""
    expected = os.environ.get("CRON_SECRET", "").strip()
    if not expected:
        raise HTTPException(status_code=500, detail="CRON_SECRET not configured")
    if not x_cron_secret or x_cron_secret.strip() != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing cron secret")


# ============================================================
# Endpoint 1: Daily news fetch + classification
# ============================================================

@router.post("/fetch-news", response_model=CronResponse)
async def cron_fetch_news(x_cron_secret: Optional[str] = Header(default=None)) -> CronResponse:
    """
    Daily job: fetch headlines from GNews + NewsAPI, classify with AI,
    save top 20 (with one star) to Supabase.

    Idempotent on source_url — duplicate URLs are skipped by the UNIQUE constraint.
    Delegates to services.news_pipeline.run_news_refresh (shared with the
    /news/headlines self-heal path).

    Triggered daily at ~06:00 IST by GitHub Actions (.github/workflows/daily-news.yml).
    """
    verify_cron_secret(x_cron_secret)

    try:
        result = run_news_refresh()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"News refresh failed: {type(e).__name__}: {e}")

    return CronResponse(
        status=result.get("status", "ok"),
        message=result.get("message", "News refreshed"),
        details={k: v for k, v in result.items() if k not in ("status", "message")},
    )


# ============================================================
# Endpoint 2: Daily cleanup of old headlines
# ============================================================

@router.post("/cleanup", response_model=CronResponse)
async def cron_cleanup(x_cron_secret: Optional[str] = Header(default=None)) -> CronResponse:
    """
    Delete headlines (and cascading briefs) older than 14 days.

    The ON DELETE CASCADE on gd_briefs.headline_id ensures associated
    briefs are deleted automatically.

    Should be called once daily via the scheduler at ~3am IST.
    """
    verify_cron_secret(x_cron_secret)

    supabase = get_supabase_client()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=14)).isoformat()

    try:
        # Get count before deletion (for response)
        count_res = supabase.table("news_headlines") \
            .select("id", count="exact", head=True) \
            .lt("published_at", cutoff) \
            .execute()
        to_delete = count_res.count or 0

        if to_delete == 0:
            return CronResponse(
                status="ok",
                message="No headlines older than 14 days to delete",
                details={"deleted": 0, "cutoff": cutoff}
            )

        # Delete
        supabase.table("news_headlines") \
            .delete() \
            .lt("published_at", cutoff) \
            .execute()

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Cleanup failed: {type(e).__name__}: {e}"
        )

    return CronResponse(
        status="ok",
        message=f"Deleted {to_delete} headlines (and cascaded briefs) older than 14 days",
        details={"deleted": to_delete, "cutoff": cutoff}
    )


# ============================================================
# Endpoint: Daily schedule cron — fills 7-day buffer
#
# Called by the scheduler every day at 6 AM IST.
# Idempotent — safe to call multiple times.
# ============================================================

@router.post("/schedule-daily", response_model=CronResponse)
async def cron_schedule_daily(x_cron_secret: Optional[str] = Header(default=None)) -> CronResponse:
    """
    Generate today's daily case + guesstimate via the AI generator and write
    them to daily_schedule (today-only, idempotent). Falls back to existing
    active cases if generation fails, so the daily surface is never empty.
    """
    verify_cron_secret(x_cron_secret)

    try:
        result = fill_daily_schedule()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Schedule fill failed: {type(e).__name__}: {e}"
        )

    return CronResponse(
        status=result.get("status", "ok"),
        message=result.get("message", "Daily schedule updated"),
        details=result,
    )
