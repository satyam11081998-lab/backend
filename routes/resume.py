"""
Resume Lab routes — AI bullet refine / generate / fit. Pro-gated, rate-limited.
"""

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional

from services.supabase_client import get_supabase_client
from services.auth import get_verified_user_id
from services.access_guard import assert_tier_at_least
from services.rate_limit import check_rate_limit
from services.resume_ai import refine_bullet, generate_bullets, fit_bullet, rebuild_resume, ResumeAIError

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


def _guard(authorization: Optional[str], bucket: str):
    supabase = get_supabase_client()
    uid = get_verified_user_id(supabase, authorization)
    check_rate_limit(f"{bucket}:{uid}", max_calls=20, window_seconds=60)
    assert_tier_at_least(supabase, uid, "pro")


def _cap(n: Optional[int]) -> int:
    try:
        v = int(n or 120)
    except (TypeError, ValueError):
        v = 120
    return max(40, min(v, MAX_CHARS_CAP))


@router.post("/refine-bullet", response_model=OptionsResponse)
async def refine(body: RefineRequest, authorization: Optional[str] = Header(default=None)) -> OptionsResponse:
    _guard(authorization, "resume_refine")
    try:
        opts = refine_bullet(body.bullet, body.domain or "", _cap(body.max_chars))
    except ResumeAIError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return OptionsResponse(options=[Option(**o) for o in opts])


@router.post("/generate-bullets", response_model=OptionsResponse)
async def generate(body: GenerateRequest, authorization: Optional[str] = Header(default=None)) -> OptionsResponse:
    _guard(authorization, "resume_generate")
    try:
        opts = generate_bullets(body.role or "", body.task or "", body.result or "", body.domain or "", body.count or 3, _cap(body.max_chars))
    except ResumeAIError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return OptionsResponse(options=[Option(**o) for o in opts])


@router.post("/fit-bullet", response_model=OptionsResponse)
async def fit(body: FitRequest, authorization: Optional[str] = Header(default=None)) -> OptionsResponse:
    _guard(authorization, "resume_fit")
    try:
        opts = fit_bullet(body.bullet, _cap(body.max_chars))
    except ResumeAIError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return OptionsResponse(options=[Option(**o) for o in opts])


class RebuildRequest(BaseModel):
    text: str


@router.post("/rebuild")
async def rebuild(body: RebuildRequest, authorization: Optional[str] = Header(default=None)):
    _guard(authorization, "resume_rebuild")
    try:
        data = rebuild_resume(body.text)
    except ResumeAIError as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"data": data}
