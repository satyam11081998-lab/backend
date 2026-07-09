"""
Resume Lab routes — AI bullet refine / generate / fit. Pro-gated, rate-limited.
"""

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional

from services.supabase_client import get_supabase_client
from services.auth import get_verified_user_id
from services.rate_limit import check_rate_limit
from services.ai_usage import assert_daily_budget
from services.access_guard import effective_tier
from services.resume_ai import refine_bullet, generate_bullets, fit_bullet, generate_points, rebuild_resume, ResumeAIError

router = APIRouter(prefix="/resume", tags=["resume"])

MAX_CHARS_CAP = 160


class Option(BaseModel):
    text: str
    chars: int
    rationale: str


class OptionsResponse(BaseModel):
    options: List[Option]


class RefineRequest(BaseModel):
    bullet: str
    domain: Optional[str] = ""
    max_chars: Optional[int] = 120


class GenerateRequest(BaseModel):
    role: Optional[str] = ""
    task: Optional[str] = ""
    result: Optional[str] = ""
    domain: Optional[str] = ""
    count: Optional[int] = 3
    max_chars: Optional[int] = 120


class FitRequest(BaseModel):
    bullet: str
    max_chars: Optional[int] = 120


class PointRequest(BaseModel):
    achievement: str
    domain: Optional[str] = ""
    max_chars: Optional[int] = 120
    count: Optional[int] = 3
    instructions: Optional[str] = ""


class PointResponse(BaseModel):
    options: List[Option] = []
    clarify: Optional[str] = None


def _guard(authorization: Optional[str], bucket: str):
    supabase = get_supabase_client()
    uid = get_verified_user_id(supabase, authorization)
    # Bullet Lab is free for any signed-in user; rate-limit still applies.
    check_rate_limit(f"{bucket}:{uid}", max_calls=20, window_seconds=60)
    assert_daily_budget()  # global spend backstop
    
    tier = effective_tier(supabase, uid)
    if tier != "pro":
        trial = supabase.table("feature_trials").select("uses").eq("user_id", uid).eq("feature", "cv_pointer_lab").execute()
        uses = trial.data[0]["uses"] if trial.data else 0
        if uses >= 2:
            raise HTTPException(403, "You have used your free preview. Upgrade to Pro for unlimited access.")
        return uid, tier, uses
    return uid, tier, 0


def _record_use(uid: str, tier: str, uses: int) -> None:
    if tier != "pro":
        supabase = get_supabase_client()
        supabase.table("feature_trials").upsert({
            "user_id": uid,
            "feature": "cv_pointer_lab",
            "uses": uses + 1
        }, on_conflict="user_id,feature").execute()


def _cap(n: Optional[int]) -> int:
    try:
        v = int(n or 120)
    except (TypeError, ValueError):
        v = 120
    return max(40, min(v, MAX_CHARS_CAP))


@router.post("/point", response_model=PointResponse)
async def point(body: PointRequest, authorization: Optional[str] = Header(default=None)) -> PointResponse:
    """Achievement -> strict-fit one-line bullets (95-100% of max_chars, never over), or a
    single clarifying question when the achievement is too vague."""
    uid, tier, uses = _guard(authorization, "resume_point")
    try:
        result = generate_points(
            body.achievement or "", body.domain or "", _cap(body.max_chars),
            body.count or 3, body.instructions or "", user_id=uid,
        )
    except ResumeAIError as e:
        raise HTTPException(status_code=502, detail=str(e))
    _record_use(uid, tier, uses)
    if result.get("clarify"):
        return PointResponse(options=[], clarify=result["clarify"])
    return PointResponse(options=[Option(**o) for o in result["options"]])


@router.post("/refine-bullet", response_model=OptionsResponse)
async def refine(body: RefineRequest, authorization: Optional[str] = Header(default=None)) -> OptionsResponse:
    uid, tier, uses = _guard(authorization, "resume_refine")
    try:
        opts = refine_bullet(body.bullet, body.domain or "", _cap(body.max_chars), user_id=uid)
    except ResumeAIError as e:
        raise HTTPException(status_code=502, detail=str(e))
    _record_use(uid, tier, uses)
    return OptionsResponse(options=[Option(**o) for o in opts])


@router.post("/generate-bullets", response_model=OptionsResponse)
async def generate(body: GenerateRequest, authorization: Optional[str] = Header(default=None)) -> OptionsResponse:
    uid, tier, uses = _guard(authorization, "resume_generate")
    try:
        opts = generate_bullets(body.role or "", body.task or "", body.result or "", body.domain or "", body.count or 3, _cap(body.max_chars), user_id=uid)
    except ResumeAIError as e:
        raise HTTPException(status_code=502, detail=str(e))
    _record_use(uid, tier, uses)
    return OptionsResponse(options=[Option(**o) for o in opts])


@router.post("/fit-bullet", response_model=OptionsResponse)
async def fit(body: FitRequest, authorization: Optional[str] = Header(default=None)) -> OptionsResponse:
    uid, tier, uses = _guard(authorization, "resume_fit")
    try:
        opts = fit_bullet(body.bullet, _cap(body.max_chars), user_id=uid)
    except ResumeAIError as e:
        raise HTTPException(status_code=502, detail=str(e))
    _record_use(uid, tier, uses)
    return OptionsResponse(options=[Option(**o) for o in opts])


class RebuildRequest(BaseModel):
    text: str


@router.post("/rebuild")
async def rebuild(body: RebuildRequest, authorization: Optional[str] = Header(default=None)):
    uid, tier, uses = _guard(authorization, "resume_rebuild")
    try:
        data = rebuild_resume(body.text, user_id=uid)
    except ResumeAIError as e:
        raise HTTPException(status_code=502, detail=str(e))
    _record_use(uid, tier, uses)
    return {"data": data}
