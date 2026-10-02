"""Plans: the one-time free interview ("trial") and Ultra.

Nothing here takes money. MECE's Razorpay flow owns payments; until Ultra is a MECE plan, it is
given by an admin (an access grant of type "ultra") or arrives as tier "ultra" on the identity
(forward-compatible: no II change is needed when MECE adds the tier).

  trial  one interview, ever, per account: `plans.trial_minutes` long, any mode, voice or text,
         full report. Offered to everyone only when `plans.trial_open` is on; until then only to
         accounts an admin gives a "trial" grant (to try the experience). Started = used; an
         interview still in progress may be finished; the report stays readable forever.
  ultra  every mode and length, voice, full reports, up to `plans.ultra_monthly_interviews`
         interviews started per 30 days (fair use: a voice interview costs real money).

The plans page is shown only to people who can use Interview Intelligence (or used their free
interview) while `plans.visibility` is "access" — it is not public.
"""

from __future__ import annotations

import uuid
from datetime import timedelta
from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..db.models import AuditLog, InterviewSession, utcnow
from . import flags

TRIAL = "trial"
ULTRA = "ultra"
LIVE_STATES = ("active", "paused")


def trial_state(db: Session, user_id: uuid.UUID) -> dict:
    """{started: bool, in_progress: bool, session_id, prepared: int}. Counts deleted interviews too:
    deleting your free interview does not give it back."""
    started = db.execute(
        select(InterviewSession.id, InterviewSession.status)
        .where(InterviewSession.user_id == user_id, InterviewSession.started_at.is_not(None))
        .order_by(InterviewSession.started_at)).all()
    prepared = db.execute(select(func.count()).select_from(InterviewSession)
                          .where(InterviewSession.user_id == user_id)).scalar() or 0
    live = next((sid for sid, status in started if status in LIVE_STATES), None)
    return {"started": bool(started), "in_progress": live is not None and len(started) == 1,
            "session_id": str(started[0][0]) if started else None, "prepared": int(prepared)}


def trial_allows_use(db: Session, user_id: uuid.UUID) -> bool:
    """May this trial account use II right now: before its one interview, or to finish it."""
    st = trial_state(db, user_id)
    return not st["started"] or st["in_progress"]


def started_last_30d(db: Session, user_id: uuid.UUID) -> int:
    since = utcnow() - timedelta(days=30)
    return int(db.execute(select(func.count()).select_from(InterviewSession)
                          .where(InterviewSession.user_id == user_id, InterviewSession.started_at >= since)).scalar() or 0)


def has_history(db: Session, user_id: uuid.UUID) -> bool:
    return db.execute(select(InterviewSession.id).where(InterviewSession.user_id == user_id,
                                                        InterviewSession.deleted_at.is_(None)).limit(1)).first() is not None


def voice_engine_for(db: Session, decision) -> str:
    """The voice engine this account's calls use. A free interview can run on a cheaper engine
    (`plans.trial_voice_engine`) than everyone else; "same" = the global `voice.engine`."""
    f = flags.all_flags(db)
    trial_engine = f.get("plans.trial_voice_engine", "same")
    if decision is not None and decision.via == TRIAL and trial_engine != "same":
        return trial_engine
    return f.get("voice.engine", "realtime")


def config(db: Session) -> dict:
    f = flags.all_flags(db)
    return {
        "visibility": f["plans.visibility"],
        "trial_open": bool(f["plans.trial_open"]),
        "trial_minutes": int(f["plans.trial_minutes"]),
        "trial_max_prepared": int(f["plans.trial_max_prepared"]),
        "ultra_price_inr": int(f["plans.ultra_price_inr"]),
        "ultra_monthly_interviews": int(f["plans.ultra_monthly_interviews"]),
        "trial_voice_engine": f["plans.trial_voice_engine"],
    }


def visible(db: Session, decision, user_id: uuid.UUID) -> bool:
    """Who may see the plans page: admins, anyone II lets in, and anyone who used a free interview."""
    if config(db)["visibility"] == "off" and not decision.is_admin:
        return False
    if decision.is_admin or decision.allowed:
        return True
    return decision.via_hint == TRIAL and has_history(db, user_id)


def interest_count(db: Session, plan: str = ULTRA) -> int:
    return int(db.execute(select(func.count(func.distinct(AuditLog.actor_user_id)))
                          .where(AuditLog.action == "plans.interest", AuditLog.target_id == plan)).scalar() or 0)


def already_interested(db: Session, user_id: uuid.UUID, plan: str = ULTRA) -> bool:
    return db.execute(select(AuditLog.id).where(AuditLog.action == "plans.interest", AuditLog.target_id == plan,
                                                AuditLog.actor_user_id == user_id).limit(1)).first() is not None


def summary(db: Session, decision, user_id: uuid.UUID) -> dict:
    """What the app needs to know about the caller's plan (also returned by /v1/me)."""
    cfg = config(db)
    out: dict = {"via": decision.via, "visible": visible(db, decision, user_id), "trial": None, "ultra": None}
    if decision.via == TRIAL or decision.via_hint == TRIAL:
        st = trial_state(db, user_id)
        out["trial"] = {"minutes": cfg["trial_minutes"], "available": not st["started"],
                        "in_progress": st["in_progress"], "used": st["started"] and not st["in_progress"],
                        "session_id": st["session_id"]}
    if decision.via == ULTRA:
        used = started_last_30d(db, user_id)
        out["ultra"] = {"monthly_interviews": cfg["ultra_monthly_interviews"], "started_last_30d": used,
                        "left": max(0, cfg["ultra_monthly_interviews"] - used)}
    return out


def public_plans(db: Session, decision, user_id: uuid.UUID, *, pro_price_inr: Optional[int] = None) -> dict:
    cfg = config(db)
    return {
        "preview": cfg["visibility"] == "access",
        "currency": "INR",
        "trial": {"minutes": cfg["trial_minutes"], "open_to_everyone": cfg["trial_open"]},
        "ultra": {"price_inr": cfg["ultra_price_inr"], "monthly_interviews": cfg["ultra_monthly_interviews"],
                  "interested": already_interested(db, user_id)},
        "you": summary(db, decision, user_id),
    }
