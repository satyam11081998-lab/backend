"""
Usage / quota route.

GET /usage/ai-quota — the caller's remaining voice minutes + OCR images for today,
used by the frontend to show "≈X min voice left" / "Y scans left" and to disable
the mic/camera gracefully once a tier limit is hit.
"""

from typing import Optional
from fastapi import APIRouter, Header

from services.supabase_client import get_supabase_client
from services.auth import get_verified_user_id
from services.rate_limit import check_rate_limit
from services.ai_usage import get_ai_input_quota

router = APIRouter(prefix="/usage", tags=["usage"])


@router.get("/ai-quota")
async def ai_quota(authorization: Optional[str] = Header(default=None)):
    supabase = get_supabase_client()
    uid = get_verified_user_id(supabase, authorization)
    check_rate_limit(f"quota:{uid}", max_calls=60, window_seconds=60)
    return get_ai_input_quota(supabase, uid)
