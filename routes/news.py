
"""
News and GD briefs routes.

Three endpoints:
1. GET  /news/headlines              — list today's curated headlines (with star)
2. POST /news/briefs/{headline_id}   — generate brief for a specific headline
3. GET  /news/briefs/{headline_id}   — fetch existing brief (no AI call)

Headlines are pre-populated by the cron job (see routes/cron.py).
Briefs are generated on-demand when users click a headline,
then cached forever (shared across users — briefs aren't personalized).
"""

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
from services.supabase_client import get_supabase_client
from services.brief_generator import generate_brief, BriefGenerationError
from services.abstract_gd_generator import generate_abstract_brief, AbstractBriefError
from services.auth import get_verified_user_id
from services.access_guard import assert_tier_at_least, effective_tier
from services.rate_limit import check_rate_limit
from services.ai_usage import assert_daily_budget
from services.news_pipeline import ensure_fresh_headlines
from datetime import datetime, timedelta, timezone


router = APIRouter(prefix="/news", tags=["news"])


# ============================================================
# Response models — what we send to the frontend
# ============================================================

class HeadlineResponse(BaseModel):
    """One headline in the list view."""
    id: str
    title: str
    description: Optional[str]
    thumbnail_url: Optional[str]
    source_url: str
    source_name: str
    published_at: str
    keywords: List[str]
    category: str
    is_star: bool
    has_brief: bool  # whether a brief has already been generated for this headline


class HeadlinesListResponse(BaseModel):
    """The full list returned by GET /headlines."""
    headlines: List[HeadlineResponse]
    count: int


class BriefResponse(BaseModel):
    """A generated GD brief — full detail view."""
    id: str
    headline_id: str
    headline_title: str
    headline_source_name: str
    headline_source_url: str
    headline_thumbnail_url: Optional[str]
    summary: str
    gd_type: str
    likely_questions: List[str]
    smart_angles: List[str]
    data_points: List[str]
    opening_lines: List[str]
    counter_arguments: List[str]
    closing_lines: List[str]
    created_at: str


# ============================================================
# Endpoint 1: List today's curated headlines
# ============================================================

@router.get("/headlines", response_model=HeadlinesListResponse)
async def list_headlines(
    authorization: Optional[str] = Header(default=None),
) -> HeadlinesListResponse:
    """
    Return today's curated headlines, sorted by:
    - Star headline first (is_star = true)
    - Then descending by gd_worthiness_score

    Frontend uses this to render the Inshorts-style list view.
    Visible to all signed-in users (free included); only brief generation/viewing is Lite/Pro.
    """
    supabase = get_supabase_client()
    # The NEWS LIST is visible to every signed-in user (free included) — seeing
    # the day's GD-worthy headlines is free. GENERATING / VIEWING a brief stays
    # Lite/Pro (gated on the two /briefs/* endpoints below). Auth + rate-limit only.
    _uid = get_verified_user_id(supabase, authorization)
    check_rate_limit(f"headlines:{_uid}", max_calls=30, window_seconds=60)

    # Self-heal: if the freshest stored headline is >24h old (or none exist),
    # refresh before returning so the user never lands on stale news. Best-effort —
    # a news-API hiccup must never break the read path.
    try:
        ensure_fresh_headlines()
    except Exception as e:
        print(f"Warning: self-heal refresh failed: {e}")

    # Show the LATEST news first (not the highest-scored from the whole retention
    # window): star pinned -> newest published -> score as tiebreaker. Window to
    # the last 3 days; fall back to most-recent-regardless if that window is empty.
    window_start = (datetime.now(timezone.utc) - timedelta(days=3)).isoformat()

    def _query_headlines(windowed: bool):
        q = supabase.table("news_headlines").select("*")
        if windowed:
            q = q.gte("fetched_at", window_start)
        return q.order("is_star", desc=True) \
            .order("published_at", desc=True) \
            .order("gd_worthiness_score", desc=True) \
            .limit(20) \
            .execute()

    try:
        headlines_res = _query_headlines(windowed=True)
        if not (headlines_res.data or []):
            headlines_res = _query_headlines(windowed=False)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch headlines: {type(e).__name__}: {e}"
        )
    
    raw_headlines = headlines_res.data or []
    
    if not raw_headlines:
        return HeadlinesListResponse(headlines=[], count=0)

    # Deduplicate headlines by normalized title — different sources often report
    # the same story, producing near-identical rows. Keep the one with the
    # highest gd_worthiness_score (star trumps non-star).
    seen_titles: dict[str, dict] = {}
    for h in raw_headlines:
        norm = h["title"].strip().lower()
        existing = seen_titles.get(norm)
        if existing is None:
            seen_titles[norm] = h
        else:
            # Prefer: star > non-star, then higher score, then newer
            h_star = h.get("is_star", False)
            e_star = existing.get("is_star", False)
            if (h_star and not e_star) or \
               (h_star == e_star and (h.get("gd_worthiness_score") or 0) > (existing.get("gd_worthiness_score") or 0)):
                seen_titles[norm] = h
    raw_headlines = list(seen_titles.values())
    
    # For each headline, check if a brief already exists
    headline_ids = [h["id"] for h in raw_headlines]
    
    try:
        briefs_res = supabase.table("gd_briefs") \
            .select("headline_id") \
            .in_("headline_id", headline_ids) \
            .execute()
    except Exception as e:
        print(f"Warning: Failed to fetch brief existence: {e}")
        briefs_res = type("obj", (), {"data": []})()  # empty fallback
    
    briefs_with_headlines = {b["headline_id"] for b in (briefs_res.data or []) if b.get("headline_id")}
    
    tier = effective_tier(supabase, _uid)
    headlines_response = [
        HeadlineResponse(
            id=h["id"],
            title=h["title"],
            description=h.get("description"),
            thumbnail_url=h.get("thumbnail_url"),
            source_url="" if tier == "free" else h["source_url"],
            source_name=h["source_name"],
            published_at=h["published_at"],
            keywords=h.get("keywords") or [],
            category=h.get("category") or "other",
            is_star=h.get("is_star", False),
            has_brief=h["id"] in briefs_with_headlines,
        )
        for h in raw_headlines
    ]
    
    return HeadlinesListResponse(headlines=headlines_response, count=len(headlines_response))


# ============================================================
# Helper: check brief access for Free tier
# ============================================================

def check_brief_access(supabase, user_id: str, headline_id: str) -> None:
    tier = effective_tier(supabase, user_id)
    if tier in ("lite", "pro"):
        return
        
    try:
        unlock_res = supabase.table("gd_brief_unlocks").select("headline_id").eq("user_id", user_id).execute()
    except Exception:
        raise HTTPException(500, "Failed to check brief unlocks")
        
    unlocks = unlock_res.data or []
    if len(unlocks) > 0:
        if unlocks[0]["headline_id"] == headline_id:
            return
        raise HTTPException(403, "You have already used your 1 free brief on another headline. Upgrade to Lite for unlimited access.")
        
    try:
        supabase.table("gd_brief_unlocks").upsert({"user_id": user_id, "headline_id": headline_id}, on_conflict="user_id,headline_id").execute()
    except Exception:
        pass
        
    try:
        unlock_res = supabase.table("gd_brief_unlocks").select("headline_id").eq("user_id", user_id).execute()
    except Exception:
        pass
    unlocks = unlock_res.data or []
    if len(unlocks) > 1:
        supabase.table("gd_brief_unlocks").delete().eq("user_id", user_id).eq("headline_id", headline_id).execute()
        raise HTTPException(403, "You have already used your 1 free brief on another headline.")


# ============================================================
# Endpoint 2: Generate brief for a headline (triggers AI)
# ============================================================

@router.post("/briefs/{headline_id}", response_model=BriefResponse)
async def generate_brief_for_headline(
    headline_id: str,
    authorization: Optional[str] = Header(default=None),
) -> BriefResponse:
    """
    Generate a GD brief for the given headline.
    If a brief already exists, returns the cached version (no AI call).
    Otherwise, calls OpenAI, saves to DB, and returns.
    
    Cost: ~₹2-4 per new brief generation. Free for cached.
    """
    supabase = get_supabase_client()
    
    # GD briefs are a Lite+ feature — verify the caller and their tier.
    _uid = get_verified_user_id(supabase, authorization)
    check_rate_limit(f"brief:{_uid}", max_calls=10, window_seconds=60)
    check_brief_access(supabase, _uid, headline_id)

    # Step 1: Check if brief already exists (cache hit = no AI call)
    try:
        existing_res = supabase.table("gd_briefs") \
            .select("*") \
            .eq("headline_id", headline_id) \
            .limit(1) \
            .execute()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to check existing brief: {type(e).__name__}: {e}"
        )

    existing_data = (existing_res.data or [None])[0] if existing_res and existing_res.data else None

    # Step 2: Fetch the headline (need it for both cache miss AND response)
    try:
        headline_res = supabase.table("news_headlines") \
            .select("*") \
            .eq("id", headline_id) \
            .limit(1) \
            .execute()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch headline: {type(e).__name__}: {e}"
        )
    
    if not headline_res or not headline_res.data:
        raise HTTPException(status_code=404, detail=f"Headline not found: {headline_id}")

    headline = headline_res.data[0]

    # Cache hit: return existing brief
    if existing_data:
        b = existing_data
        return BriefResponse(
            id=b["id"],
            headline_id=headline_id,
            headline_title=headline["title"],
            headline_source_name=headline["source_name"],
            headline_source_url=headline["source_url"],
            headline_thumbnail_url=headline.get("thumbnail_url"),
            summary=b["summary"],
            gd_type=b.get("gd_type") or "Case-based",
            likely_questions=b.get("likely_questions") or [],
            smart_angles=b.get("smart_angles") or b.get("points_for") or [],
            data_points=b.get("data_points") or [],
            opening_lines=b.get("opening_lines") or [b.get("how_to_open")] if b.get("how_to_open") else [],
            counter_arguments=b.get("counter_arguments") or b.get("points_against") or [],
            closing_lines=b.get("closing_lines") or [b.get("how_to_close")] if b.get("how_to_close") else [],
            created_at=b["created_at"],
        )
    
    # Cache miss: generate new brief via AI
    assert_daily_budget()  # global spend backstop (cache hits above are unaffected)
    try:
        brief = generate_brief(
            headline_title=headline["title"],
            headline_description=headline.get("description"),
            headline_source=headline["source_name"],
            headline_keywords=headline.get("keywords") or [],
            headline_category=headline.get("category") or "other",
        )
    except BriefGenerationError as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate brief: {str(e)}"
        )
    
    # Save the generated brief to Supabase. Using insert and handling UniqueViolation (23505)
    # to gracefully resolve concurrent cache-miss races, bypassing the 42P10 partial index issue.
    try:
        insert_res = supabase.table("gd_briefs").insert({
            "headline_id": headline_id,
            "topic": headline["title"],
            "summary": brief["summary"],
            "gd_type": brief["gd_type"],
            "likely_questions": brief["likely_questions"],
            "smart_angles": brief["smart_angles"],
            "data_points": brief["data_points"],
            "opening_lines": brief["opening_lines"],
            "counter_arguments": brief["counter_arguments"],
            "closing_lines": brief["closing_lines"],
            "source_url": headline["source_url"],
            # Legacy columns kept for backward compat
            "points_for": brief["smart_angles"],
            "points_against": brief["counter_arguments"],
            "how_to_open": brief["opening_lines"][0] if brief["opening_lines"] else "",
            "how_to_close": brief["closing_lines"][0] if brief["closing_lines"] else "",
        }).execute()
        
        if not insert_res.data or len(insert_res.data) == 0:
            raise HTTPException(status_code=500, detail="Supabase returned empty insert result")
            
        saved = insert_res.data[0]
        
    except Exception as e:
        error_str = str(e)
        if "23505" in error_str or "duplicate key" in error_str.lower() or "42P10" in error_str:
            # A concurrent request already saved this brief, fetch it instead.
            fetch_res = supabase.table("gd_briefs").select("*").eq("headline_id", headline_id).execute()
            if not fetch_res.data or len(fetch_res.data) == 0:
                raise HTTPException(status_code=500, detail="Failed to fetch existing brief after conflict")
            saved = fetch_res.data[0]
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to save brief: {type(e).__name__}: {e}"
            )
    

    return BriefResponse(
        id=saved["id"],
        headline_id=headline_id,
        headline_title=headline["title"],
        headline_source_name=headline["source_name"],
        headline_source_url=headline["source_url"],
        headline_thumbnail_url=headline.get("thumbnail_url"),
        summary=brief["summary"],
        gd_type=brief["gd_type"],
        likely_questions=brief["likely_questions"],
        smart_angles=brief["smart_angles"],
        data_points=brief["data_points"],
        opening_lines=brief["opening_lines"],
        counter_arguments=brief["counter_arguments"],
        closing_lines=brief["closing_lines"],
        created_at=saved["created_at"],
    )


# ============================================================
# Endpoint 3: Fetch existing brief (no AI call, fast read)
# ============================================================

@router.get("/briefs/{headline_id}", response_model=BriefResponse)
async def get_brief(
    headline_id: str,
    authorization: Optional[str] = Header(default=None),
) -> BriefResponse:
    """
    Fetch an existing brief. Does NOT generate a new one.
    Returns 404 if no brief exists yet for this headline.

    Used by frontend when navigating to /gd-briefs/[id] from a list item
    where has_brief=true.

    GD Briefs is a Lite+ feature. This read path was previously UNGATED, so a
    free/anon caller could pull any cached brief straight from the API. Now it
    requires a verified JWT and Lite/Pro tier, matching the POST generate path.
    """
    supabase = get_supabase_client()
    _uid = get_verified_user_id(supabase, authorization)
    check_rate_limit(f"brief_get:{_uid}", max_calls=30, window_seconds=60)
    check_brief_access(supabase, _uid, headline_id)

    try:
        brief_res = supabase.table("gd_briefs") \
            .select("*") \
            .eq("headline_id", headline_id) \
            .limit(1) \
            .execute()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch brief: {type(e).__name__}: {e}"
        )

    if not brief_res or not brief_res.data:
        raise HTTPException(status_code=404, detail="Brief not generated yet")

    try:
        headline_res = supabase.table("news_headlines") \
            .select("*") \
            .eq("id", headline_id) \
            .limit(1) \
            .execute()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch headline: {type(e).__name__}: {e}"
        )

    if not headline_res or not headline_res.data:
        raise HTTPException(status_code=404, detail="Headline not found")

    b = brief_res.data[0]
    h = headline_res.data[0]
    
    return BriefResponse(
        id=b["id"],
        headline_id=headline_id,
        headline_title=h["title"],
        headline_source_name=h["source_name"],
        headline_source_url=h["source_url"],
        headline_thumbnail_url=h.get("thumbnail_url"),
        summary=b["summary"],
        gd_type=b.get("gd_type") or "Case-based",
        likely_questions=b.get("likely_questions") or [],
        smart_angles=b.get("smart_angles") or b.get("points_for") or [],
        data_points=b.get("data_points") or [],
        opening_lines=b.get("opening_lines") or ([b["how_to_open"]] if b.get("how_to_open") else []),
        counter_arguments=b.get("counter_arguments") or b.get("points_against") or [],
        closing_lines=b.get("closing_lines") or ([b["how_to_close"]] if b.get("how_to_close") else []),
        created_at=b["created_at"],
    )


# ============================================================
# Endpoint: Abstract GD brief (on-demand, not news-based)
# ============================================================

class AbstractBriefRequest(BaseModel):
    topic: str


class AbstractBriefResponse(BaseModel):
    topic: str
    interpretations: List[str]
    idea_pool: List[str]
    lenses: List[str]
    balanced_for: List[str]
    balanced_against: List[str]
    analogies: List[str]
    sample_structure: List[str]
    pitfalls: List[str]
    opening_lines: List[str]
    closing_lines: List[str]


@router.post("/abstract-brief", response_model=AbstractBriefResponse)
async def generate_abstract_gd_brief(
    body: AbstractBriefRequest,
    authorization: Optional[str] = Header(default=None),
) -> AbstractBriefResponse:
    """
    Generate an Abstract GD brief for any abstract topic (a word/phrase/proverb).
    Teaches how to crack THIS topic and, by repetition, the method for any abstract
    topic. Lite/Pro gated, rate-limited; generated on demand (not stored).
    """
    supabase = get_supabase_client()
    _uid = get_verified_user_id(supabase, authorization)
    check_rate_limit(f"abstract_brief:{_uid}", max_calls=10, window_seconds=60)
    assert_tier_at_least(supabase, _uid, "lite")

    topic = (body.topic or "").strip()
    if not topic:
        raise HTTPException(status_code=400, detail="Topic is required.")
    if len(topic) > 160:
        raise HTTPException(status_code=400, detail="Topic is too long (max 160 chars).")

    # Cache (migration 0036): abstract briefs are deterministic-enough that the same
    # topic must never re-bill gpt-4o. Preset topic chips guarantee repeats.
    topic_key = " ".join(topic.lower().split())
    try:
        hit = supabase.table("abstract_briefs").select("brief").eq("topic_key", topic_key).maybe_single().execute()
        if hit and hit.data and hit.data.get("brief"):
            return AbstractBriefResponse(**hit.data["brief"])
    except Exception:
        pass  # cache is best-effort — fall through to generate

    assert_daily_budget()  # global spend backstop (only on a real cache miss)
    try:
        brief = generate_abstract_brief(topic)
    except AbstractBriefError as e:
        raise HTTPException(status_code=502, detail=f"Could not generate brief: {e}")

    try:
        supabase.table("abstract_briefs").upsert(
            {"topic_key": topic_key, "topic": topic, "brief": brief}, on_conflict="topic_key"
        ).execute()
    except Exception:
        pass  # caching failure must not break the response

    return AbstractBriefResponse(**brief)
