"""
Request authentication helpers.

The backend uses the Supabase service-role client (RLS bypass), so any
write MUST first prove who the caller is. We verify the Supabase access
token (JWT) the frontend forwards in the Authorization header and derive
the user id from it — never trust a user_id supplied in the request body.
"""

from typing import Optional
from fastapi import HTTPException


def get_verified_user_id(supabase, authorization: Optional[str]) -> str:
    """Validate the Bearer access token and return the authenticated user id.

    Raises HTTPException(401) if the header is missing or the token is invalid.
    """
    token = (authorization or "").replace("Bearer ", "").replace("bearer ", "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing authentication token")
    try:
        res = supabase.auth.get_user(token)
        uid = getattr(getattr(res, "user", None), "id", None)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired authentication token")
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid or expired authentication token")
    return uid
