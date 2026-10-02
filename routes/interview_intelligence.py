"""Interview Intelligence (II) — mounted at /ii inside this backend.

II is a self-contained package in ./interview-intelligence/ with its own code, database
schema (`interview_intel`, own Postgres role), prompts, model routing, admin and audit. It
imports nothing from this backend. This file is the only place the two meet: it tells II who
the caller is, using this backend's own Supabase session check, so the frontend calls
/ii/... with the same Supabase access token it already sends to every other route.

Cost and safety:
  * No extra service. AI keys are shared (II reads OPENAI_API_KEY etc. when no II_* key is set).
  * Dormant until II_DATABASE_URL is set: every /ii call answers 503 "not_configured".
  * Lazy: nothing heavy (SQLAlchemy, parsers, II's routers) is imported until the first /ii
    call, so the backend's start-up memory is unchanged until someone uses II.
  * Never breaks the backend: if II fails to start, that is logged and /ii answers 503; if it
    cannot even be mounted, that is logged and the rest of the backend starts normally.
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

log = logging.getLogger("interview_intelligence.glue")

_II_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "interview-intelligence")


def _users_row(supabase, uid: str) -> dict:
    """One users read: tier, expiry, admin, guest. Same deploy-order guard as
    services/access_guard.effective_tier_and_guest for the is_guest column."""
    cols = "subscription_tier, subscription_expires_at, is_admin, is_guest"
    try:
        res = supabase.table("users").select(cols).eq("id", uid).maybe_single().execute()
    except Exception as exc:  # noqa: BLE001 — only the missing-column case degrades
        if "is_guest" not in str(exc):
            raise
        res = supabase.table("users").select(
            "subscription_tier, subscription_expires_at, is_admin").eq("id", uid).maybe_single().execute()
    return (res.data or {}) if res else {}


def _confirmed_email(user) -> str:
    email = (getattr(user, "email", None) or "").strip().lower()
    if not email:
        return ""
    if type(user).__name__ == "_ClaimsUser":
        # LOCAL_JWT_VERIFY path: the email claim comes from a Supabase-signed token.
        return email
    # II's test access is granted by email, so only a confirmed address counts.
    confirmed = getattr(user, "email_confirmed_at", None) or getattr(user, "confirmed_at", None)
    return email if confirmed else ""


def resolve_identity(authorization: Optional[str]):
    """Authorization header -> II HostIdentity. Raises the backend's HTTPException(401) on a
    missing/invalid token; II turns that into its own error shape."""
    from interview_intelligence.host import HostIdentity
    from services.access_guard import _effective_tier_from_row
    from services.auth import get_verified_user, is_guest_user
    from services.supabase_client import get_supabase_client

    supabase = get_supabase_client()
    uid, user = get_verified_user(supabase, authorization)
    if is_guest_user(user):
        return HostIdentity(user_id=str(uid), email="", tier="free", is_guest=True)
    row = _users_row(supabase, str(uid))
    return HostIdentity(
        user_id=str(uid),
        email=_confirmed_email(user),
        tier=_effective_tier_from_row(row),
        is_admin=bool(row.get("is_admin")),
        is_guest=bool(row.get("is_guest")),
    )


def mount_interview_intelligence(app) -> None:
    try:
        if _II_DIR not in sys.path:
            sys.path.append(_II_DIR)  # append: the backend's own modules always win
        from interview_intelligence.host import mount
        mount(app, "/ii", resolve_identity)
        state = "active" if os.getenv("II_DATABASE_URL", "").strip() else "dormant (II_DATABASE_URL not set)"
        print(f"[interview-intelligence] mounted at /ii — {state}")
    except Exception:  # noqa: BLE001 — II must never take the backend down
        log.exception("[interview-intelligence] could not be mounted; /ii is unavailable")
