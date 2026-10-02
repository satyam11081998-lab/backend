"""Request plumbing shared by every router."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator, Optional, Tuple

from fastapi import Header
from sqlalchemy.orm import Session

from ..access import policy, rate_limit
from ..access.policy import AccessDecision
from ..ai.runner import collect_runs
from ..auth.assertion import Principal, extract_bearer, principal_from_host, verify_assertion
from ..db.models import User
from ..db.session import db_session


def principal(authorization: Optional[str] = Header(default=None)) -> Principal:
    from ..host import identity_resolver
    resolver = identity_resolver()
    if resolver is not None:  # host mode: the host verified its own session token
        return principal_from_host(authorization, resolver)
    return verify_assertion(extract_bearer(authorization))


@contextmanager
def unit() -> Iterator[Session]:
    """AI-run collection wraps the DB unit of work, so model-run logs are persisted after the
    request's own transaction finishes — whether it committed or rolled back."""
    with collect_runs():
        with db_session() as db:
            yield db


def use(db: Session, p: Principal, *, klass: str = "read", write: bool = True) -> Tuple[User, AccessDecision]:
    """Identity + entitlement for routes that start or continue interviews / upload documents."""
    rate_limit.check(str(p.user_id), klass)
    d = policy.require_use(db, p)
    u = policy.ensure_user(db, p, d)
    return u, d


def read_own(db: Session, p: Principal) -> Tuple[User, AccessDecision]:
    """Identity only: a user may always read their OWN history (even after Pro lapses)."""
    rate_limit.check(str(p.user_id), "read")
    d = policy.decide(db, p)
    if not d.allowed and db.get(User, p.user_id) is None:
        # PII minimisation (spec §63): someone who has never had access is not registered just
        # for looking. A transient object (never added to the session) answers the read.
        return User(id=p.user_id, email=p.email, email_lc=p.email_lc, last_tier=p.tier), d
    u = policy.ensure_user(db, p, d)
    return u, d


def admin(db: Session, p: Principal) -> User:
    rate_limit.check(str(p.user_id), "admin")
    policy.require_admin(p)
    return policy.ensure_user(db, p, policy.decide(db, p))
