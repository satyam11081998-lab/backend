"""
Broadcast targeted-practice — admin-only case/guesstimate generation for email campaigns.

  POST /broadcast/generate-options -> N distinct DRAFT options for a topic (nothing saved).
  POST /broadcast/materialize      -> save ONE chosen option as an UNLISTED case and return
                                      {case_id, ...} to drop a "Practice this ->" link into a
                                      broadcast email.

Admin-gated (users.is_admin), rate-limited, and behind the daily-budget kill switch —
the exact contract used by routes/seo.py and routes/coach.py.
"""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from services.auth import get_verified_user, is_guest_user
from services.supabase_client import get_supabase_client
from services.rate_limit import check_rate_limit
from services.ai_usage import assert_daily_budget

router = APIRouter(prefix="/broadcast", tags=["broadcast", "growth", "admin"])


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


class OptionsRequest(BaseModel):
    topic: str = ""
    kind: str = "case"          # "case" | "guesstimate"
    difficulty: str = "medium"  # easy | medium | hard
    count: int = 3


class MaterializeRequest(BaseModel):
    option: Dict[str, Any]
    topic: str = ""


@router.post("/generate-options")
async def broadcast_generate_options(body: OptionsRequest, authorization: Optional[str] = Header(default=None)):
    uid = _require_admin(authorization)
    check_rate_limit(f"broadcast:gen:{uid}", max_calls=20, window_seconds=60)
    assert_daily_budget()  # 503 if the day's AI spend is over budget
    from services.broadcast_gen import generate_options
    try:
        options = generate_options(body.topic, body.kind, body.difficulty, body.count)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Generation failed: {type(e).__name__}")
    return {"options": options, "kind": (body.kind or "case").lower()}


@router.post("/materialize")
async def broadcast_materialize(body: MaterializeRequest, authorization: Optional[str] = Header(default=None)):
    uid = _require_admin(authorization)
    check_rate_limit(f"broadcast:save:{uid}", max_calls=30, window_seconds=60)
    from services.broadcast_gen import save_option
    try:
        saved = save_option(get_supabase_client(), uid, body.option, body.topic)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Save failed: {type(e).__name__}")
    return saved
