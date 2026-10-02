"""Verification of the MECE -> II entitlement assertion (contract C10 v1).

The ONLY identity/entitlement input II accepts. Signed by the MECE Next.js server with an
Ed25519 private key II never holds. See docs/C_API_CONTRACT.md §1.
"""

from __future__ import annotations

import hashlib
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import List, Optional

import jwt
from jwt import InvalidTokenError

from ..config import get_settings
from ..errors import Forbidden, IIError, Unauthorized

ALLOWED_ALGS = ["EdDSA"]  # pinned: no 'none', no HS* (key-confusion), no RS*
ENTITLEMENT = "interview_intelligence"
CONTRACT_VERSION = 1


@dataclass(frozen=True)
class Principal:
    user_id: uuid.UUID
    email: str
    tier: str
    entitlements: List[str] = field(default_factory=list)
    mece_admin: bool = False
    jti: str = ""
    issued_at: int = 0
    expires_at: int = 0

    @property
    def email_lc(self) -> str:
        return (self.email or "").strip().lower()

    @property
    def pro_entitled(self) -> bool:
        return ENTITLEMENT in self.entitlements


_GENERIC = "Your session could not be verified. Refresh the page and try again."


def _load_keys():
    from cryptography.hazmat.primitives.serialization import load_pem_public_key

    keys = {}
    for kid, pem in get_settings().assertion_public_keys.items():
        try:
            keys[kid] = load_pem_public_key(pem.encode("utf-8"))
        except Exception:
            continue
    return keys


def extract_bearer(authorization: Optional[str]) -> str:
    raw = (authorization or "").strip()
    if not raw.lower().startswith("bearer "):
        raise Unauthorized("Missing authentication token.", code="missing_token")
    token = raw[7:].strip()
    if not token or token.count(".") != 2:
        raise Unauthorized(_GENERIC, code="invalid_token")
    return token


def verify_assertion(token: str, *, now: Optional[int] = None) -> Principal:
    s = get_settings()
    keys = _load_keys()
    if not keys:
        raise Unauthorized("Interview Intelligence is not configured to accept sign-ins yet.",
                           code="assertion_keys_missing")
    try:
        header = jwt.get_unverified_header(token)
    except InvalidTokenError:
        raise Unauthorized(_GENERIC, code="invalid_token")
    if header.get("alg") not in ALLOWED_ALGS:
        raise Unauthorized(_GENERIC, code="invalid_alg")
    kid = header.get("kid")
    candidates = [keys[kid]] if kid and kid in keys else list(keys.values())

    claims = None
    last_err: Optional[Exception] = None
    for key in candidates:
        try:
            claims = jwt.decode(
                token,
                key,
                algorithms=ALLOWED_ALGS,
                audience=s.assertion_audience,
                issuer=s.assertion_issuer,
                leeway=s.assertion_leeway_s,
                options={"require": ["exp", "iat", "sub", "iss", "aud", "jti"]},
            )
            break
        except jwt.ExpiredSignatureError:
            raise Unauthorized("Your session expired. Refresh the page.", code="token_expired")
        except InvalidTokenError as e:
            last_err = e
            continue
    if claims is None:
        raise Unauthorized(_GENERIC, code="invalid_token") from last_err

    if claims.get("ver") != CONTRACT_VERSION:
        raise Unauthorized(_GENERIC, code="unsupported_version")
    iat = int(claims["iat"])
    exp = int(claims["exp"])
    if exp - iat > s.assertion_max_lifetime_s:
        raise Unauthorized(_GENERIC, code="lifetime_too_long")
    current = int(now if now is not None else time.time())
    if iat > current + s.assertion_leeway_s:
        raise Unauthorized(_GENERIC, code="issued_in_future")
    try:
        uid = uuid.UUID(str(claims["sub"]))
    except ValueError:
        raise Unauthorized(_GENERIC, code="invalid_subject")
    tier = str(claims.get("tier") or "free")
    if tier not in {"free", "lite", "pro"}:
        tier = "free"
    ent = claims.get("ent") or []
    if not isinstance(ent, list):
        ent = []
    return Principal(
        user_id=uid,
        email=str(claims.get("email") or ""),
        tier=tier,
        entitlements=[str(e) for e in ent],
        mece_admin=bool(claims.get("adm") is True),
        jti=str(claims.get("jti")),
        issued_at=iat,
        expires_at=exp,
    )


# ---------------------------------------------------------------------------------------
# Host mode (interview_intelligence/host.py): the host process verifies its own session token
# and hands II an identity. Same Principal, same access policy downstream.
# ---------------------------------------------------------------------------------------
_host_cache: "OrderedDict[str, tuple[float, Principal]]" = OrderedDict()
_host_lock = threading.Lock()
HOST_CACHE_TTL_S = 60.0
HOST_CACHE_MAX = 2000


def principal_from_host(authorization: Optional[str], resolver) -> Principal:
    token = (authorization or "").strip()
    if not token.lower().startswith("bearer ") or not token[7:].strip():
        raise Unauthorized("Missing authentication token.", code="missing_token")
    key = hashlib.sha256(token.encode()).hexdigest()
    now = time.time()
    with _host_lock:
        hit = _host_cache.get(key)
        if hit and now - hit[0] < HOST_CACHE_TTL_S:
            return hit[1]
    try:
        ident = resolver(authorization)
    except IIError:
        raise
    except Exception as e:  # the host's own 401/403 types (e.g. fastapi.HTTPException)
        status = getattr(e, "status_code", 401)
        if status == 403:
            raise Forbidden(str(getattr(e, "detail", "")) or "Access denied.", code="forbidden")
        raise Unauthorized("Your session could not be verified. Refresh the page and try again.",
                           code="invalid_token")
    if ident is None:
        raise Unauthorized(_GENERIC, code="invalid_token")
    if ident.is_guest:
        raise Forbidden("Create a free account to use Interview Intelligence.", code="guest")
    try:
        uid = uuid.UUID(str(ident.user_id))
    except ValueError:
        raise Unauthorized(_GENERIC, code="invalid_subject")
    tier = ident.tier if ident.tier in {"free", "lite", "pro"} else "free"
    p = Principal(user_id=uid, email=(ident.email or "").strip().lower(), tier=tier,
                  entitlements=[ENTITLEMENT] if tier == "pro" else [], mece_admin=bool(ident.is_admin),
                  jti=key[:16], issued_at=int(now), expires_at=int(now + HOST_CACHE_TTL_S))
    with _host_lock:
        _host_cache[key] = (now, p)
        while len(_host_cache) > HOST_CACHE_MAX:
            _host_cache.popitem(last=False)
    return p


def reset_host_cache() -> None:
    with _host_lock:
        _host_cache.clear()
