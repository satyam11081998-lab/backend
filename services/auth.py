"""
Request authentication helpers.

The backend uses the Supabase service-role client (RLS bypass), so any
write MUST first prove who the caller is. We verify the Supabase access
token (JWT) the frontend forwards in the Authorization header and derive
the user id from it — never trust a user_id supplied in the request body.
"""

import base64
import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from typing import Optional
from fastapi import HTTPException


# --- Optional local JWT fast-path -------------------------------------------
# supabase.auth.get_user(token) is a NETWORK round-trip to Supabase on EVERY
# authenticated request. ONE spoken interview turn makes 3-5 of them
# (/transcribe + /attempts/messages + /speak x N), so that hop sits squarely on
# the hot path of the latency-critical feature.
#
# When LOCAL_JWT_VERIFY is truthy we first try to verify the access token's
# signature LOCALLY (no network) and read id/email/anon from the claims. On ANY
# miss -- flag off, no key material, bad signature, expired, missing claim -- we
# fall through to the exact network call used today. So this can never reject a
# token the old path accepted, and never accepts one whose signature it could
# not verify.
#
# Signing schemes, tried in order:
#   1) Asymmetric (JWKS): modern Supabase, needs no secret; keys fetched from
#      <SUPABASE_URL>/auth/v1/.well-known/jwks.json and cached in-process.
#   2) Legacy HS256: set SUPABASE_JWT_SECRET (Dashboard > Settings > API > JWT).
#
# OPT-IN (default off) because local verify checks signature+expiry but, unlike
# get_user(), cannot see a server-side revocation or a just-deleted user until
# the token expires (~1h). Flip LOCAL_JWT_VERIFY off to restore server-checked
# auth instantly.
_LOCAL_JWT = os.getenv("LOCAL_JWT_VERIFY", "").strip().lower() in ("1", "true", "yes", "on")
_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "").strip()
_SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip().rstrip("/")
_JWT_AUD = (os.getenv("SUPABASE_JWT_AUD", "authenticated").strip() or "authenticated")

_jwks_client = None
_jwks_tried = False


def _jwks():
    global _jwks_client, _jwks_tried
    if _jwks_client is not None or _jwks_tried:
        return _jwks_client
    _jwks_tried = True
    if not _SUPABASE_URL:
        return None
    try:
        import jwt
        _jwks_client = jwt.PyJWKClient(f"{_SUPABASE_URL}/auth/v1/.well-known/jwks.json")
    except Exception:
        _jwks_client = None
    return _jwks_client


class _ClaimsUser:
    """Minimal stand-in for the Supabase user object, built from verified JWT
    claims. Exposes exactly the attributes callers read (.id, .is_anonymous,
    .email -- see routes/copilot.py); any other attribute reads as None rather
    than raising, so an unexpected access degrades to 'absent', never a 500."""
    __slots__ = ("id", "is_anonymous", "email")

    def __init__(self, claims: dict):
        self.id = claims.get("sub")
        self.is_anonymous = bool(claims.get("is_anonymous", False))
        self.email = claims.get("email")

    def __getattr__(self, _name):  # only for names not in __slots__
        return None


def _verify_local(token: str):
    """Return a _ClaimsUser if the token verifies locally, else None. Never raises."""
    if not _LOCAL_JWT:
        return None
    try:
        import jwt
    except Exception:
        return None
    claims = None
    cli = _jwks()
    if cli is not None:
        try:
            key = cli.get_signing_key_from_jwt(token).key
            claims = jwt.decode(
                token, key, algorithms=["ES256", "RS256"],
                audience=_JWT_AUD, options={"require": ["exp", "sub"]},
            )
        except Exception:
            claims = None
    if claims is None and _JWT_SECRET:
        try:
            claims = jwt.decode(
                token, _JWT_SECRET, algorithms=["HS256"],
                audience=_JWT_AUD, options={"require": ["exp", "sub"]},
            )
        except Exception:
            claims = None
    if not claims or not claims.get("sub"):
        return None
    return _ClaimsUser(claims)


# --- Verified-token cache ------------------------------------------------------
# Even with LOCAL_JWT_VERIFY off, a token Supabase has ALREADY verified does not
# need a second Oregon->Tokyo round trip seconds later: a live interview sends
# the same token on every turn, TTS sentence and transcript save. A successful
# get_user() result is remembered for AUTH_CACHE_TTL_SECONDS (default 60), and
# never past the token's own expiry. Only successes are cached -- a bad token is
# re-checked (and rejected) every time -- and guest (anonymous) users are never
# cached, so a guest who signs up counts as a full account at once. Keyed by a
# hash, so raw tokens are not kept. Trade-off, same kind as LOCAL_JWT_VERIFY but bounded to the TTL: a
# server-side revocation takes up to that long to bite. AUTH_CACHE_TTL_SECONDS=0
# turns the cache off.
def _ttl_env(name: str, default: float) -> float:
    try:
        return max(0.0, float(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


_AUTH_CACHE_TTL = _ttl_env("AUTH_CACHE_TTL_SECONDS", 60.0)
_AUTH_CACHE_MAX = 5000
_auth_cache: "OrderedDict[str, tuple]" = OrderedDict()
_auth_lock = threading.Lock()


def _token_key(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _token_exp(token: str) -> Optional[float]:
    """The token's `exp` claim, read WITHOUT verification -- used only to stop a
    cache entry outliving the token, never to accept anything."""
    try:
        payload = token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        exp = json.loads(base64.urlsafe_b64decode(payload.encode("ascii"))).get("exp")
        return float(exp) if exp is not None else None
    except Exception:
        return None


def _cache_get(token: str):
    if _AUTH_CACHE_TTL <= 0:
        return None
    key = _token_key(token)
    with _auth_lock:
        hit = _auth_cache.get(key)
        if hit is None:
            return None
        expires_at, uid, user = hit
        if time.time() >= expires_at:
            _auth_cache.pop(key, None)
            return None
        _auth_cache.move_to_end(key)
        return uid, user


def _cache_put(token: str, uid: str, user) -> None:
    if _AUTH_CACHE_TTL <= 0:
        return
    now = time.time()
    expires_at = now + _AUTH_CACHE_TTL
    exp = _token_exp(token)
    if exp is not None:
        expires_at = min(expires_at, exp)
    if expires_at <= now:
        return
    key = _token_key(token)
    with _auth_lock:
        _auth_cache[key] = (expires_at, uid, user)
        _auth_cache.move_to_end(key)
        while len(_auth_cache) > _AUTH_CACHE_MAX:
            _auth_cache.popitem(last=False)


def get_verified_user(supabase, authorization: Optional[str]):
    """Validate the Bearer access token and return (uid, user_object).

    Raises HTTPException(401) if the header is missing or the token is invalid.
    """
    token = (authorization or "").replace("Bearer ", "").replace("bearer ", "").strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing authentication token")
    # Fast path: verify locally (opt-in via LOCAL_JWT_VERIFY). Any miss falls
    # through to the network call below, so behavior is identical when local
    # verify is off or cannot verify the token.
    local = _verify_local(token)
    if local is not None and local.id:
        return local.id, local
    cached = _cache_get(token)
    if cached is not None:
        return cached
    try:
        res = supabase.auth.get_user(token)
        user = getattr(res, "user", None)
        uid = getattr(user, "id", None)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired authentication token")
    if not uid:
        raise HTTPException(status_code=401, detail="Invalid or expired authentication token")
    if getattr(user, "is_anonymous", False) is not True:
        _cache_put(token, uid, user)
    return uid, user


def get_verified_user_id(supabase, authorization: Optional[str]) -> str:
    """Validate the Bearer access token and return the authenticated user id.

    Raises HTTPException(401) if the header is missing or the token is invalid.
    """
    uid, _ = get_verified_user(supabase, authorization)
    return uid


def is_guest_user(user) -> bool:
    return getattr(user, 'is_anonymous', False) is True
