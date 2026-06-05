"""
Cron-triggered routes for the GD briefs system.

Two endpoints called by external scheduler (cron-job.org):
1. POST /cron/fetch-news    — daily: fetch + classify headlines
2. POST /cron/cleanup       — daily: delete headlines older than 14 days

Both require the X-Cron-Secret header to match CRON_SECRET env var.
This prevents random users from triggering expensive operations.
"""

import os
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Optional
from services.supabase_client import get_supabase_client
from services.news_fetcher import fetch_all_headlines
from services.daily_scheduler import fill_daily_schedule
from services.headline_classifier import (
    classify_headlines,
    filter_top_headlines,
    ClassificationError,
)


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
    
    Triggered daily at ~06:00 IST by GitHub Actions (.github/workflows).
    """
    verify_cron_secret(x_cron_secret)
    
    # Step 1: Fetch and classify in a loop until we have at least 10 highly-scored headlines
    top_headlines = []
    page = 1
    max_results = 50
    total_fetched = 0
    total_classified = 0
    
    while len(top_headlines) < 10 and page <= 5:
        try:
            raw_headlines = fetch_all_headlines(max_results=max_results, page=page)
            if not raw_headlines:
                break
                
            total_fetched += len(raw_headlines)
            
            classified = classify_headlines(raw_headlines)
            total_classified += len(classified)
            
            # Filter this batch for minimum score of 75
            batch_top = filter_top_headlines(classified, top_n=50, min_score=75)
            top_headlines.extend(batch_top)
            
            # Deduplicate by URL
            seen_urls = set()
            unique_top = []
            for h in top_headlines:
                if h["source_url"] not in seen_urls:
                    seen_urls.add(h["source_url"])
                    unique_top.append(h)
            top_headlines = unique_top
            
            if len(top_headlines) >= 10:
                break
                
            # Prepare for next iteration
            page += 1
            max_results = 20
            
        except Exception as e:
            if page == 1:
                raise HTTPException(status_code=500, detail=f"Failed to fetch news on first try: {type(e).__name__}: {e}")
            else:
                print(f"Warning: Fetch loop failed on page {page}: {e}")
                break
    
    # Trim to Top 20 max to avoid saving too many
    top_headlines = sorted(top_headlines, key=lambda h: (not h["is_star"], -h["gd_worthiness_score"]))[:20]
    
    if not top_headlines:
        return CronResponse(
            status="warning",
            message="No headlines survived classification or none fetched",
            details={"fetched": total_fetched, "classified": total_classified, "saved": 0}
        )
    
    # Step 4: Insert into Supabase (skip duplicates via unique constraint on source_url)
    supabase = get_supabase_client()
    saved_count = 0
    skipped_count = 0
    
    for h in top_headlines:
        try:
            supabase.table("news_headlines").insert({
                "title": h["title"],
                "description": h["description"],
                "thumbnail_url": h["thumbnail_url"],
                "source_url": h["source_url"],
                "source_name": h["source_name"],
                "published_at": h["published_at"],
                "gd_worthiness_score": h["gd_worthiness_score"],
                "is_star": h["is_star"],
                "keywords": h["keywords"],
                "category": h["category"],
            }).execute()
            saved_count += 1
        except Exception as e:
            err_str = str(e).lower()
            # Duplicate URL (unique constraint) — expected, just skip
            if "duplicate" in err_str or "unique" in err_str:
                skipped_count += 1
            else:
                print(f"Failed to save headline '{h['title'][:60]}': {type(e).__name__}: {e}")
                skipped_count += 1
    
    # Step 5: If we have a star headline, ensure only ONE is_star=true exists for today
    # (defensive — in case duplicates were already in DB)
    try:
        # Clear stars on headlines older than 24 hours
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        supabase.table("news_headlines") \
            .update({"is_star": False}) \
            .lt("fetched_at", cutoff) \
            .execute()
    except Exception as e:
        print(f"Warning: failed to demote old stars: {e}")
    
    return CronResponse(
        status="ok",
        message=f"Fetched {total_fetched}, saved {saved_count}, skipped {skipped_count}",
        details={
            "fetched": total_fetched,
            "classified": total_classified,
            "considered_top": len(top_headlines),
            "saved": saved_count,
            "skipped_duplicates": skipped_count,
            "pages_fetched": page,
        }
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
    
    Should be called once daily via cron-job.org at ~3am IST.
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
# Called by cron-job.org every day at 6 AM IST.
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