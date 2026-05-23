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
    
    Should be called once daily via cron-job.org at ~9am IST.
    """
    verify_cron_secret(x_cron_secret)
    
    # Step 1: Fetch from both news APIs
    try:
        raw_headlines = fetch_all_headlines()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch news: {type(e).__name__}: {e}"
        )
    
    if not raw_headlines:
        return CronResponse(
            status="warning",
            message="No headlines fetched from any source",
            details={"fetched": 0, "saved": 0}
        )
    
    # Step 2: Classify with AI (single batched call)
    try:
        classified = classify_headlines(raw_headlines)
    except ClassificationError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Classification failed: {str(e)}"
        )
    
    # Step 3: Keep top 20 (star always included)
    top_headlines = filter_top_headlines(classified, top_n=20)
    
    if not top_headlines:
        return CronResponse(
            status="warning",
            message="No headlines survived classification",
            details={"fetched": len(raw_headlines), "classified": len(classified), "saved": 0}
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
        message=f"Fetched {len(raw_headlines)}, saved {saved_count}, skipped {skipped_count}",
        details={
            "fetched": len(raw_headlines),
            "classified": len(classified),
            "considered_top": len(top_headlines),
            "saved": saved_count,
            "skipped_duplicates": skipped_count,
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