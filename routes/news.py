
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

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from services.supabase_client import get_supabase_client
from services.brief_generator import generate_brief, BriefGenerationError


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
async def list_headlines() -> HeadlinesListResponse:
    """
    Return today's curated headlines, sorted by:
    - Star headline first (is_star = true)
    - Then descending by gd_worthiness_score
    
    Frontend uses this to render the Inshorts-style list view.
    """
    supabase = get_supabase_client()
    
    try:
        # Fetch headlines from last 14 days (cron deletes older)
        headlines_res = supabase.table("news_headlines") \
            .select("*") \
            .order("is_star", desc=True) \
            .order("gd_worthiness_score", desc=True) \
            .order("published_at", desc=True) \
            .limit(20) \
            .execute()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch headlines: {type(e).__name__}: {e}"
        )
    
    raw_headlines = headlines_res.data or []
    
    if not raw_headlines:
        return HeadlinesListResponse(headlines=[], count=0)
    
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
    
    headlines_response = [
        HeadlineResponse(
            id=h["id"],
            title=h["title"],
            description=h.get("description"),
            thumbnail_url=h.get("thumbnail_url"),
            source_url=h["source_url"],
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
# Endpoint 2: Generate brief for a headline (triggers AI)
# ============================================================

@router.post("/briefs/{headline_id}", response_model=BriefResponse)
async def generate_brief_for_headline(headline_id: str) -> BriefResponse:
    """
    Generate a GD brief for the given headline.
    If a brief already exists, returns the cached version (no AI call).
    Otherwise, calls OpenAI, saves to DB, and returns.
    
    Cost: ~₹2-4 per new brief generation. Free for cached.
    """
    supabase = get_supabase_client()
    
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
    
    # Save the generated brief to Supabase
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
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save brief: {type(e).__name__}: {e}"
        )
    
    if not insert_res.data or len(insert_res.data) == 0:
        raise HTTPException(status_code=500, detail="Supabase returned empty insert result")
    
    saved = insert_res.data[0]
    
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
async def get_brief(headline_id: str) -> BriefResponse:
    """
    Fetch an existing brief. Does NOT generate a new one.
    Returns 404 if no brief exists yet for this headline.
    
    Used by frontend when navigating to /gd-briefs/[id] from a list item
    where has_brief=true.
    """
    supabase = get_supabase_client()
    
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