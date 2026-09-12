"""
Growth Agent routes — programmatic SEO generation (admin only).

  GET  /seo/candidates -> fresh, GD-worthy headlines not yet turned into a page.
  POST /seo/generate   -> generate ONE grounded SEO draft from a headline (or an
                          explicit topic), self-critique it, and store it as a
                          DRAFT in seo_pages. Never publishes — an admin approves
                          in /admin/growth, which flips status to 'published'.

Admin-gated with the same contract as routes/agentic.py & routes/coach.py.
The daily-budget kill switch is enforced before any model call.
"""

from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from services.auth import get_verified_user, is_guest_user
from services.supabase_client import get_supabase_client
from services.rate_limit import check_rate_limit
from services.ai_usage import assert_daily_budget

router = APIRouter(prefix="/seo", tags=["seo", "growth", "admin"])


def _require_admin(authorization: Optional[str]) -> str:
    supabase = get_supabase_client()
    uid, user_obj = get_verified_user(supabase, authorization)
    if is_guest_user(user_obj):
        raise HTTPException(status_code=403, detail="Admins only")
    try:
        res = supabase.table("users").select("is_admin").eq("id", uid).single().execute()
        is_admin = bool((res.data or {}).get("is_admin"))
    except Exception:
        raise HTTPException(status_code=403, detail="Admins only")
    if not is_admin:
        raise HTTPException(status_code=403, detail="Admins only")
    return uid


class GenerateRequest(BaseModel):
    headline_id: Optional[str] = None
    topic: Optional[str] = None


@router.get("/candidates")
async def seo_candidates(authorization: Optional[str] = Header(default=None)):
    _require_admin(authorization)
    from services.growth import list_candidate_headlines
    return {"candidates": list_candidate_headlines(get_supabase_client(), limit=12)}


@router.post("/generate")
async def seo_generate(body: GenerateRequest, authorization: Optional[str] = Header(default=None)):
    uid = _require_admin(authorization)
    check_rate_limit(f"seo:gen:{uid}", max_calls=20, window_seconds=60)
    assert_daily_budget()  # 503 if the day's AI spend is over budget

    from services.growth import generate_seo_page
    try:
        draft = generate_seo_page(
            get_supabase_client(), uid,
            headline_id=(body.headline_id or None),
            topic=(body.topic or None),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Generation failed: {type(e).__name__}")
    return draft
