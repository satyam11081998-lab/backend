"""Session service: create (with the 2-active rule), view, list, re-attempt."""

from __future__ import annotations

import uuid
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..access import flags
from ..access.audit import audit
from ..access.policy import AccessDecision
from ..ai.runner import spend_today_usd
from ..db.models import Document, InterviewSession, Report, User
from ..documents.service import get_owned
from ..errors import NotFound, TooMany, Unavailable, Unprocessable
from ..jobs.queue import enqueue
from .lifecycle import assert_slot_available, lock_user, sessions_today, sweep_user
from .modes import MODES, InterviewConfig

PUBLIC_STATUSES = ("created", "uploading", "analyzing", "ready", "active", "paused", "completed", "abandoned",
                   "expired", "failed")


def _enqueue_assessments(db: Session, ids: List[uuid.UUID]) -> None:
    for sid in ids:
        s = db.get(InterviewSession, sid)
        if s is not None:
            s.assessment_status = "pending"
        enqueue(db, "assess_session", {"session_id": str(sid)}, dedupe_key=f"assess:{sid}", max_attempts=3)


def create_session(db: Session, user: User, decision: AccessDecision, *, cv_document_id: uuid.UUID,
                   jd_document_id: uuid.UUID, config: dict, source_session_id: Optional[uuid.UUID] = None) -> InterviewSession:
    f = flags.all_flags(db)
    try:
        cfg = InterviewConfig(**(config or {}))
    except Exception as e:  # pydantic ValidationError -> clean 422
        raise Unprocessable(f"Invalid interview settings: {e}".split("\n")[0][:300], code="bad_config")
    if cfg.duration_minutes not in f["limits.allowed_durations"]:
        raise Unprocessable(f"Duration must be one of {f['limits.allowed_durations']} minutes.", code="bad_duration")
    if cfg.voice and not f["voice.enabled"]:
        cfg.voice = False
    if cfg.mode == "technical_deep_dive" and not f["technical.advanced_mode"]:
        raise Unprocessable("Technical deep dive is not available right now.", code="mode_unavailable")

    lock_user(db, user.id)
    _enqueue_assessments(db, sweep_user(db, user.id))

    budget = float(f["limits.daily_budget_usd"] or 0)
    if budget and spend_today_usd() >= budget:
        raise Unavailable("Interview Intelligence has reached today's capacity. Please try again tomorrow.",
                          code="capacity")
    assert_slot_available(db, user.id, int(f["limits.max_active_sessions"]))
    per_day = int(f["limits.max_sessions_per_day_test"] if decision.via in ("test_grant", "admin")
                  else f["limits.max_sessions_per_day"])
    if per_day and sessions_today(db, user.id) >= per_day:
        raise TooMany(f"You've started {per_day} interviews in the last 24 hours. Please come back a little later.",
                      code="daily_limit")

    cv = get_owned(db, user.id, cv_document_id)
    jd = get_owned(db, user.id, jd_document_id)
    if cv.kind != "cv" or jd.kind != "jd":
        raise Unprocessable("Choose a CV and a job description.", code="wrong_document_kind")
    for d, label in ((cv, "CV"), (jd, "job description")):
        if d.parse_status != "parsed":
            raise Unprocessable(f"Your {label} couldn't be read. {d.parse_error or 'Please upload it again.'}",
                                code="document_unreadable")

    src = None
    if source_session_id:
        src = db.execute(select(InterviewSession).where(InterviewSession.id == source_session_id,
                                                        InterviewSession.user_id == user.id)).scalar_one_or_none()
        if src is None:
            raise NotFound("Previous interview not found.")
    if cfg.target_competencies:
        # Targets reach the model prompts as competency names, so only known ids are accepted:
        # canonical library ids, or role-specific (rs:) ids from the source interview's model.
        from ..db.models import InterviewBlueprint
        from ..role_taxonomy.library import library
        allowed_rs: set = set()
        if src is not None:
            bp = db.execute(select(InterviewBlueprint).where(InterviewBlueprint.session_id == src.id)).scalar_one_or_none()
            if bp is not None:
                allowed_rs = {c.get("competency_id") for c in (bp.competency_model or {}).get("competencies", [])}
        lib = library()
        bad = [t for t in cfg.target_competencies if not (lib.competency(t) or (t.startswith("rs:") and t in allowed_rs))]
        if bad:
            raise Unprocessable("Unknown competency to target: " + ", ".join(b[:40] for b in bad[:3]), code="bad_target")
    if cfg.mode == "weakness_targeting" and not cfg.target_competencies:
        targets: List[str] = []
        if src is not None:
            rep = db.execute(select(Report).where(Report.session_id == src.id)).scalar_one_or_none()
            if rep is not None:
                targets = (rep.report.get("reattempt") or {}).get("suggested_competencies", [])
        if not targets:
            raise Unprocessable("Weakness targeting needs a previous interview with identified development areas, "
                                "or competencies to target.", code="no_targets")
        cfg.target_competencies = targets[:4]

    sess = InterviewSession(user_id=user.id, status="created", cv_document_id=cv.id, jd_document_id=jd.id,
                            source_session_id=src.id if src else None, config=cfg.model_dump(), mode=cfg.mode,
                            difficulty=cfg.difficulty, duration_minutes=cfg.duration_minutes,
                            company_name=cfg.company_name[:200])
    db.add(sess)
    db.flush()
    enqueue(db, "prepare_session", {"session_id": str(sess.id)}, dedupe_key=f"prepare:{sess.id}", max_attempts=3)
    audit(db, "session.create", actor_user_id=user.id, actor_email=user.email, target_type="session",
          target_id=sess.id, meta={"mode": cfg.mode, "difficulty": cfg.difficulty, "duration": cfg.duration_minutes,
                                   "via": decision.via})
    return sess


def view(db: Session, sess: InterviewSession, *, detail: bool = True) -> dict:
    out = {
        "id": str(sess.id), "status": sess.status, "status_reason": sess.status_reason or None,
        "mode": sess.mode, "mode_label": MODES[sess.mode].label if sess.mode in MODES else sess.mode,
        "difficulty": sess.difficulty, "duration_minutes": sess.duration_minutes,
        "role_title": sess.role_title, "role_family": sess.role_family, "company_name": sess.company_name,
        "created_at": sess.created_at.isoformat(),
        "started_at": sess.started_at.isoformat() if sess.started_at else None,
        "ended_at": sess.ended_at.isoformat() if sess.ended_at else None,
        "ended_reason": sess.ended_reason or None, "active_seconds": sess.active_seconds,
        "assessment_status": sess.assessment_status, "assessment_confidence": sess.assessment_confidence or None,
        "source_session_id": str(sess.source_session_id) if sess.source_session_id else None,
    }
    if detail:
        out["config"] = sess.config
        out["cv_document_id"] = str(sess.cv_document_id)
        out["jd_document_id"] = str(sess.jd_document_id)
        if sess.status not in ("created", "uploading", "analyzing", "failed"):
            out["pre_interview_summary"] = sess.pre_interview_summary
    return out


def list_for(db: Session, user_id: uuid.UUID) -> List[InterviewSession]:
    return list(db.execute(select(InterviewSession).where(InterviewSession.user_id == user_id,
                                                          InterviewSession.deleted_at.is_(None))
                           .order_by(InterviewSession.created_at.desc()).limit(50)).scalars())
