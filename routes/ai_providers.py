"""
Admin AI-provider toggle (Admin-only).

Surfaces services/ai_providers.FEATURES so an admin can switch each feature
between OpenAI / Groq (and Google, for TTS) at runtime — no redeploy. Backed by
the ai_provider_settings table; the resolver picks changes up within its short
cache TTL (~30s).

- GET  /admin/ai-providers        -> every feature, its choices, and what's live now.
- POST /admin/ai-providers        -> set one feature's provider. Body: {feature, provider}.

Scoring is intentionally NOT toggleable (its only allowed provider is OpenAI),
so an attempt to move it is rejected by set_provider's allow-list check.
"""

from typing import Optional

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from services.auth import get_verified_user, is_guest_user
from services.supabase_client import get_supabase_client
from services.ai_providers import list_features, set_provider

router = APIRouter(prefix="/admin/ai-providers", tags=["admin"])


def _require_admin(authorization: Optional[str]) -> str:
    """Bearer token -> admin user id. 401 unauthenticated, 403 not an admin.
    Same contract as routes/decks.py so admin gating stays consistent."""
    supabase = get_supabase_client()
    uid, user_obj = get_verified_user(supabase, authorization)   # 401 if missing/invalid
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


class SetProviderRequest(BaseModel):
    feature: str
    provider: str


@router.get("")
async def get_ai_providers(authorization: Optional[str] = Header(default=None)):
    """Everything the admin toggle UI renders: per-feature label, allowed
    providers, code default, current live provider, and the model each maps to."""
    _require_admin(authorization)
    return {"features": list_features()}


@router.post("")
async def post_ai_provider(
    body: SetProviderRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Set one feature's provider. Validated against the feature's allow-list in
    set_provider (so 'scoring' can never be moved off OpenAI, and no unknown
    provider is accepted). Cache is refreshed immediately inside set_provider."""
    uid = _require_admin(authorization)
    try:
        set_provider(body.feature, body.provider, updated_by=uid)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update provider: {e}")
    return {"features": list_features()}
