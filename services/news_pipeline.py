"""
News refresh pipeline — single source of truth for fetching + classifying +
storing GD-worthy headlines.

Why this module exists
----------------------
The fetch logic used to live inline in `routes/cron.py`. It is now here so two
callers can share it:
  1. `POST /cron/fetch-news`  — the scheduled daily run (GitHub Actions, 6 AM IST).
  2. `GET /news/headlines`    — a self-heal guard: if the freshest stored headline
     is >24h old, the read path triggers a refresh before returning, with a
     retry so the user is "double sure" they get fresh news.

`run_news_refresh()` is intentionally idempotent (insert dedupes on the
`source_url` UNIQUE constraint), so calling it from both places is safe.
"""

import threading
from datetime import datetime, timedelta, timezone
from typing import Optional

from services.supabase_client import get_supabase_client
from services.news_fetcher import fetch_all_headlines
from services.headline_classifier import (
    classify_headlines,
    filter_top_headlines,
)

# Self-heal throttle: at most one auto-refresh per cooldown window, so a burst of
# reads against stale data doesn't trigger a stampede of (paid) fetch+classify runs.
_AUTO_REFRESH_COOLDOWN = timedelta(minutes=15)
_last_auto_refresh: Optional[datetime] = None
_auto_refresh_lock = threading.Lock()

# Default staleness threshold — the freshest headline must be newer than this.
STALE_AFTER_HOURS = 24


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    """Parse a Supabase timestamptz string into an aware UTC datetime."""
    if not value:
        return None
    try:
        # Supabase returns e.g. "2026-06-14T05:30:00+00:00" or "...Z"
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def latest_fetched_at(supabase) -> Optional[datetime]:
    """Return the most recent `fetched_at` across all stored headlines, or None."""
    try:
        res = (
            supabase.table("news_headlines")
            .select("fetched_at")
            .order("fetched_at", desc=True)
            .limit(1)
            .execute()
        )
        rows = res.data or []
        if not rows:
            return None
        return _parse_iso(rows[0].get("fetched_at"))
    except Exception:
        return None


def headlines_are_stale(supabase, max_age_hours: int = STALE_AFTER_HOURS) -> bool:
    """True if there are no headlines, or the freshest is older than max_age_hours."""
    latest = latest_fetched_at(supabase)
    if latest is None:
        return True
    return (datetime.now(timezone.utc) - latest) > timedelta(hours=max_age_hours)


def run_news_refresh() -> dict:
    """
    Fetch headlines from GNews + NewsAPI, classify with AI, save the top ~20
    (one star) to Supabase. Idempotent on source_url. Returns a details dict.

    Moved verbatim from the old `/cron/fetch-news` body so the scheduled job and
    the self-heal path behave identically.
    """
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

            page += 1
            max_results = 20

        except Exception as e:
            if page == 1:
                raise RuntimeError(f"Failed to fetch news on first try: {type(e).__name__}: {e}")
            else:
                print(f"Warning: Fetch loop failed on page {page}: {e}")
                break

    # Trim to Top 20 (star first, then by score)
    top_headlines = sorted(
        top_headlines, key=lambda h: (not h["is_star"], -h["gd_worthiness_score"])
    )[:20]

    if not top_headlines:
        return {
            "status": "warning",
            "message": "No headlines survived classification or none fetched",
            "fetched": total_fetched,
            "classified": total_classified,
            "saved": 0,
        }

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
            if "duplicate" in err_str or "unique" in err_str:
                skipped_count += 1
            else:
                print(f"Failed to save headline '{h['title'][:60]}': {type(e).__name__}: {e}")
                skipped_count += 1

    # Demote stars older than 24h so the star always reflects a fresh story.
    try:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
        supabase.table("news_headlines") \
            .update({"is_star": False}) \
            .lt("fetched_at", cutoff) \
            .execute()
    except Exception as e:
        print(f"Warning: failed to demote old stars: {e}")

    return {
        "status": "ok",
        "message": f"Fetched {total_fetched}, saved {saved_count}, skipped {skipped_count}",
        "fetched": total_fetched,
        "classified": total_classified,
        "considered_top": len(top_headlines),
        "saved": saved_count,
        "skipped_duplicates": skipped_count,
        "pages_fetched": page,
    }


def ensure_fresh_headlines(max_age_hours: int = STALE_AFTER_HOURS, retries: int = 1) -> dict:
    """
    Self-heal: if stored headlines are stale (or absent), run a refresh before the
    caller reads them. Retries once more if still stale after the first refresh, so
    a single transient miss doesn't leave the user on yesterday's news.

    Best-effort and throttled: never raises (the read path must still work even if
    the news APIs are down), and triggers at most once per cooldown window.
    """
    global _last_auto_refresh
    supabase = get_supabase_client()

    if not headlines_are_stale(supabase, max_age_hours):
        return {"refreshed": False, "reason": "fresh"}

    now = datetime.now(timezone.utc)
    with _auto_refresh_lock:
        if _last_auto_refresh and (now - _last_auto_refresh) < _AUTO_REFRESH_COOLDOWN:
            return {"refreshed": False, "reason": "cooldown"}
        _last_auto_refresh = now

    attempts = 0
    last_result: Optional[dict] = None
    # attempts: 1 initial + `retries` extra = the "double sure" re-fetch loop
    while attempts <= retries:
        attempts += 1
        try:
            last_result = run_news_refresh()
        except Exception as e:
            last_result = {"status": "error", "message": f"{type(e).__name__}: {e}"}
        if not headlines_are_stale(supabase, max_age_hours):
            return {"refreshed": True, "attempts": attempts, "result": last_result}

    return {"refreshed": True, "attempts": attempts, "result": last_result, "still_stale": True}
