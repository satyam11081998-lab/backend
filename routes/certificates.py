"""
Certificate routes, admin-only AI drafting of certificate copy.

There is exactly one endpoint and it is gated three ways: a valid Supabase
access token, users.is_admin on the caller, and the shared daily AI budget.
Guests are rejected outright.
"""

from typing import List, Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from services.ai_usage import assert_daily_budget
from services.auth import get_verified_user, is_guest_user
from services.certificate_ai import CertificateAIError, draft_certificate_copy
from services.rate_limit import check_rate_limit
from services.supabase_client import get_supabase_client

router = APIRouter(prefix="/certificates", tags=["certificates"])


class DraftRequest(BaseModel):
    work_notes: str
    recipient_program: Optional[str] = ""
    target_roles: Optional[List[str]] = None
    duration_label: Optional[str] = ""


class Alternatives(BaseModel):
    role_title: List[str] = []
    scope_line: List[str] = []


class DraftResponse(BaseModel):
    role_title: str
    scope_line: str
    alternatives: Alternatives


def _require_admin(authorization: Optional[str]) -> str:
    """Bearer token -> admin user id. 401 unauthenticated, 403 not an admin."""
    supabase = get_supabase_client()
    uid, user_obj = get_verified_user(supabase, authorization)
    if is_guest_user(user_obj):
        raise HTTPException(status_code=403, detail="Admins only")

    try:
        res = supabase.table("users").select("is_admin").eq("id", uid).single().execute()
        is_admin = bool((res.data or {}).get("is_admin"))
    except Exception:
        # Fail CLOSED. A lookup failure must not hand out an admin-only endpoint.
        raise HTTPException(status_code=403, detail="Admins only")

    if not is_admin:
        raise HTTPException(status_code=403, detail="Admins only")
    return uid


@router.post("/draft", response_model=DraftResponse)
async def draft(
    body: DraftRequest,
    authorization: Optional[str] = Header(default=None),
) -> DraftResponse:
    uid = _require_admin(authorization)
    # Generous for a human issuing certificates, tight enough that a leaked
    # admin cookie cannot be turned into a free GPT-4o endpoint.
    check_rate_limit(f"certificate_draft:{uid}", 30, 3600)
    assert_daily_budget()

    try:
        result = draft_certificate_copy(
            body.work_notes or "",
            recipient_program=body.recipient_program or "",
            target_roles=body.target_roles or [],
            duration_label=body.duration_label or "",
            user_id=uid,
        )
    except CertificateAIError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return DraftResponse(
        role_title=result["role_title"],
        scope_line=result["scope_line"],
        alternatives=Alternatives(**result["alternatives"]),
    )
