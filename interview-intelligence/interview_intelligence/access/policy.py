"""Access policy (docs/J_ADMIN_ACCESS.md).

allowed = flag(ii.enabled) AND ( admin OR active grant (test | ultra | trial*) OR tier "ultra"
                                 OR (pro entitlement AND flag(ii.enabled_for_pro))
                                 OR (flag(plans.trial_open) AND trial*) )
  * trial = one interview per account, ever (access/plans.py): allowed before it starts and while
    it is in progress; afterwards the account can still read its history and report.

The entitlement comes ONLY from the signed assertion (auth/assertion.py). No request body
or query field can influence it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..auth.assertion import Principal
from ..config import get_settings
from ..db.models import AccessGrant, User, utcnow
from ..errors import Forbidden
from . import flags
from .audit import audit


@dataclass(frozen=True)
class AccessDecision:
    allowed: bool
    via: Optional[str]  # admin | test_grant | ultra | pro | trial
    reason: str
    is_admin: bool
    read_only: bool  # may read own history but not start/continue interviews
    via_hint: Optional[str] = None  # the plan a refused account is on (e.g. "trial" once it is used)


def is_ii_admin(p: Principal) -> bool:
    s = get_settings()
    if p.email_lc and p.email_lc in s.admin_emails:
        return True
    return bool(s.trust_mece_admin and p.mece_admin)


def active_grant(db: Session, email_lc: str) -> Optional[AccessGrant]:
    if not email_lc:
        return None
    g = db.execute(select(AccessGrant).where(AccessGrant.email_lc == email_lc)).scalar_one_or_none()
    if g is None or g.status != "enabled":
        return None
    if g.expires_at is not None:
        exp = g.expires_at if g.expires_at.tzinfo else g.expires_at.replace(tzinfo=utcnow().tzinfo)
        if exp < utcnow():
            return None
    return g


def decide(db: Session, p: Principal) -> AccessDecision:
    f = flags.all_flags(db)
    admin = is_ii_admin(p)
    if admin:
        return AccessDecision(True, "admin", "admin", True, False)
    if not f.get("ii.enabled", True):
        return AccessDecision(False, None, "Interview Intelligence is temporarily unavailable.", False, True)
    grant = active_grant(db, p.email_lc) if f.get("admin.test_access", True) else None
    if grant is not None and grant.grant_type == "ultra":
        return AccessDecision(True, "ultra", "ultra (granted)", False, False)
    if grant is not None and grant.grant_type != "trial":
        return AccessDecision(True, "test_grant", "test access", False, False)
    if (p.tier or "").lower() == "ultra":  # a future MECE plan: no II change needed when it ships
        return AccessDecision(True, "ultra", "ultra", False, False)
    if p.pro_entitled and f.get("ii.enabled_for_pro", False):
        return AccessDecision(True, "pro", "pro", False, False)
    if grant is not None or f.get("plans.trial_open", False):
        from .plans import trial_allows_use
        if trial_allows_use(db, p.user_id):
            return AccessDecision(True, "trial", "free interview", False, False)
        return AccessDecision(False, None, "You've used your free interview. Your report stays here whenever you "
                                           "want it.", False, True, via_hint="trial")
    if p.pro_entitled:
        return AccessDecision(False, None, "Interview Intelligence is in private preview and opening to Pro soon.",
                              False, True)
    return AccessDecision(False, None, "Interview Intelligence is a Pro feature.", False, True)


def require_use(db: Session, p: Principal) -> AccessDecision:
    d = decide(db, p)
    if not d.allowed:
        raise Forbidden(d.reason, code="not_entitled")
    return d


def require_admin(p: Principal) -> None:
    if not is_ii_admin(p):
        raise Forbidden("Admin access required.", code="admin_only")


_SEEN_REFRESH = timedelta(minutes=5)


def ensure_user(db: Session, p: Principal, decision: Optional[AccessDecision] = None) -> User:
    """Upsert the II-local user record (II's own registry, never MECE's users table)."""
    u = db.get(User, p.user_id)
    now = utcnow()
    if u is None:
        u = User(id=p.user_id, email=p.email, email_lc=p.email_lc, last_tier=p.tier,
                 last_entitled=p.pro_entitled, last_access_via=decision.via if decision else None,
                 created_at=now, last_seen_at=now)
        db.add(u)
        db.flush()
        return u
    changed = (u.email_lc != p.email_lc or u.last_tier != p.tier or u.last_entitled != p.pro_entitled
               or (decision and u.last_access_via != decision.via))
    seen = u.last_seen_at if u.last_seen_at.tzinfo else u.last_seen_at.replace(tzinfo=now.tzinfo)
    if changed or now - seen > _SEEN_REFRESH:
        u.email, u.email_lc, u.last_tier, u.last_entitled = p.email, p.email_lc, p.tier, p.pro_entitled
        if decision:
            u.last_access_via = decision.via
        u.last_seen_at = now
    return u


def bootstrap_test_grants(db: Session) -> int:
    """Insert II_BOOTSTRAP_TEST_EMAILS once. Never re-enables a grant an admin disabled."""
    added = 0
    for email in get_settings().bootstrap_test_emails:
        if "@" not in email:
            continue
        exists = db.execute(select(AccessGrant).where(AccessGrant.email_lc == email)).scalar_one_or_none()
        if exists is None:
            db.add(AccessGrant(email_lc=email, status="enabled", grant_type="test",
                               note="bootstrap (II_BOOTSTRAP_TEST_EMAILS)", granted_by="system"))
            audit(db, "access_grant.bootstrap", actor_email="system", target_type="access_grant", target_id=email)
            added += 1
    return added
