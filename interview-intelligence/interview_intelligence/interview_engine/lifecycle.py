"""Explicit session state machine + the two-active-sessions rule (spec §5, docs/E §1)."""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Dict, List, Optional, Set

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import InterviewEvent, InterviewSession, User, utcnow
from ..errors import Conflict

SLOT_STATES: Set[str] = {"created", "uploading", "analyzing", "ready", "active", "paused"}
TERMINAL: Set[str] = {"completed", "abandoned", "expired", "failed"}

TRANSITIONS: Dict[str, Set[str]] = {
    "created": {"uploading", "analyzing", "ready", "failed", "abandoned", "expired"},
    "uploading": {"analyzing", "failed", "abandoned", "expired"},
    "analyzing": {"ready", "failed", "abandoned", "expired"},
    "ready": {"active", "abandoned", "expired", "failed"},
    "active": {"paused", "completed", "abandoned", "failed", "expired"},
    "paused": {"active", "completed", "abandoned", "expired"},
    "completed": set(), "abandoned": set(), "expired": set(), "failed": set(),
}

PREP_TIMEOUT = timedelta(hours=2)
READY_TIMEOUT = timedelta(hours=24)
IDLE_TO_PAUSE = timedelta(minutes=20)
PAUSED_TIMEOUT = timedelta(hours=24)


class InvalidTransition(Conflict):
    code = "invalid_transition"


def can_transition(src: str, dst: str) -> bool:
    return dst in TRANSITIONS.get(src, set())


def transition(db: Session, sess: InterviewSession, dst: str, *, reason: str = "") -> None:
    src = sess.status
    if src == dst:
        return
    if not can_transition(src, dst):
        raise InvalidTransition(f"This interview can't go from '{src}' to '{dst}'.")
    now = utcnow()
    sess.status = dst
    sess.status_reason = reason[:500]
    if dst == "active" and sess.started_at is None:
        sess.started_at = now
    if dst == "paused":
        sess.paused_at = now
    if dst == "active":
        sess.paused_at = None
    if dst in TERMINAL:
        sess.ended_at = now
    sess.last_activity_at = now
    db.add(InterviewEvent(session_id=sess.id, type="status", payload={"from": src, "to": dst, "reason": reason[:300]}))


def _aware(dt):
    return dt if dt is None or dt.tzinfo else dt.replace(tzinfo=utcnow().tzinfo)


def sweep_user(db: Session, user_id: uuid.UUID) -> List[uuid.UUID]:
    """Lazy timeout sweep for one user's slot-occupying sessions. Returns sessions whose
    assessment must be enqueued (expired with answers)."""
    from .assessment_hooks import should_assess_on_expiry  # local import avoids a cycle

    now = utcnow()
    to_assess: List[uuid.UUID] = []
    rows = db.execute(select(InterviewSession).where(InterviewSession.user_id == user_id,
                                                     InterviewSession.status.in_(SLOT_STATES))).scalars().all()
    for s in rows:
        last = _aware(s.last_activity_at) or _aware(s.created_at)
        age = now - last
        if s.status in ("created", "uploading", "analyzing") and age > PREP_TIMEOUT:
            transition(db, s, "failed", reason="Preparation timed out.")
        elif s.status == "ready" and age > READY_TIMEOUT:
            transition(db, s, "expired", reason="Not started within 24 hours.")
        elif s.status == "active" and age > IDLE_TO_PAUSE:
            transition(db, s, "paused", reason="Paused after 20 minutes without activity.")
            if age > PAUSED_TIMEOUT:
                s.ended_reason = "expired"
                transition(db, s, "expired", reason="Inactive for 24 hours.")
                if should_assess_on_expiry(db, s):
                    to_assess.append(s.id)
        elif s.status == "paused" and age > PAUSED_TIMEOUT:
            s.ended_reason = "expired"
            transition(db, s, "expired", reason="Paused for more than 24 hours.")
            if should_assess_on_expiry(db, s):
                to_assess.append(s.id)
    return to_assess


def lock_user(db: Session, user_id: uuid.UUID) -> User:
    """Serialise slot accounting per user (two concurrent 'start' clicks)."""
    u = db.execute(select(User).where(User.id == user_id).with_for_update()).scalar_one()
    return u


def active_sessions(db: Session, user_id: uuid.UUID) -> List[InterviewSession]:
    return list(db.execute(select(InterviewSession).where(InterviewSession.user_id == user_id,
                                                          InterviewSession.status.in_(SLOT_STATES),
                                                          InterviewSession.deleted_at.is_(None))
                           .order_by(InterviewSession.created_at)).scalars())


def sessions_today(db: Session, user_id: uuid.UUID) -> int:
    start = utcnow() - timedelta(hours=24)
    return int(db.execute(select(func.count()).select_from(InterviewSession)
                          .where(InterviewSession.user_id == user_id, InterviewSession.created_at >= start)).scalar() or 0)


def assert_slot_available(db: Session, user_id: uuid.UUID, max_active: int) -> None:
    act = active_sessions(db, user_id)
    if len(act) >= max_active:
        raise Conflict(
            f"You already have {len(act)} active interview sessions. Finish, end or abandon one to start another.",
            code="active_limit",
            extra={"active_sessions": [{"id": str(s.id), "status": s.status, "role_title": s.role_title,
                                        "created_at": s.created_at.isoformat()} for s in act]},
        )


def next_message_seq(db: Session, session_id: uuid.UUID) -> int:
    from ..db.models import InterviewMessage
    v = db.execute(select(func.max(InterviewMessage.seq)).where(InterviewMessage.session_id == session_id)).scalar()
    return int(v or 0) + 1


def is_terminal(status: str) -> bool:
    return status in TERMINAL


def slot_occupying(status: str) -> bool:
    return status in SLOT_STATES


def find_owned(db: Session, user_id: uuid.UUID, session_id: uuid.UUID, *, lock: bool = False) -> Optional[InterviewSession]:
    q = select(InterviewSession).where(InterviewSession.id == session_id, InterviewSession.user_id == user_id,
                                       InterviewSession.deleted_at.is_(None))
    if lock:
        q = q.with_for_update()
    return db.execute(q).scalar_one_or_none()
