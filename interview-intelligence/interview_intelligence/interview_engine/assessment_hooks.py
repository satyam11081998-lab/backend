"""When does a session that did not finish normally still get a (limited) assessment?"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import InterviewMessage, InterviewSession

MIN_SUBSTANTIVE_ANSWERS = 2


def substantive_answers(db: Session, session_id) -> int:
    rows = db.execute(select(InterviewMessage.content).where(InterviewMessage.session_id == session_id,
                                                             InterviewMessage.role == "candidate")).scalars().all()
    return sum(1 for r in rows if len((r or "").split()) >= 12)


def should_assess_on_expiry(db: Session, sess: InterviewSession) -> bool:
    return sess.started_at is not None and substantive_answers(db, sess.id) >= MIN_SUBSTANTIVE_ANSWERS
