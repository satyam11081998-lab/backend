"""Plans: Free (the one-time free interview, "trial"), Pro and Ultra.

Nothing here takes money. MECE's Razorpay flow owns payments; until Ultra is a MECE plan, it is
given by an admin (an access grant of type "ultra") or arrives as tier "ultra" on the identity
(forward-compatible: no II change is needed when MECE adds the tier).

  trial  (Free) one interview, ever, per account: `plans.trial_minutes` long (10), the everyday
         interview types, voice or text, full report. Offered to everyone only when
         `plans.trial_open` is on; until then only to accounts an admin gives a "trial" grant.
         Started = used; an interview still in progress may be finished; the report stays forever.
  pro    (only while `ii.enabled_for_pro` is on) `plans.pro_interview_minutes` (20) long,
         `plans.pro_monthly_interviews` started per 30 days, the everyday and the role-specific
         interview types. `plans.pro_limits` off = Pro gets everything (the pre-plans behaviour).
  ultra  every mode and length, the re-attempt loop, voice, full reports, up to
         `plans.ultra_monthly_interviews` interviews started per 30 days (fair use: a voice
         interview costs real money).
  Test grants and admins are not on a plan: everything, no plan caps.

Which plan first includes each interview type is MODE_PLAN below — the server enforces it on
prepare and again on start; the app only shows it.

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
PRO = "pro"
ULTRA = "ultra"
LIVE_STATES = ("active", "paused")

# The plan that first includes each interview type ("free" = every plan). The everyday interview
# is in every plan; role-specific formats come with Pro; the hardest formats (final round, the
# hiring manager, pressure) and the re-attempt loop (weakness_targeting) are Ultra.
MODE_PLAN = {
    "mixed": "free", "cv_jd": "free", "hr_behavioral": "free", "case": "free",
    "functional": "pro", "technical": "pro", "situational": "pro", "cv_deep_dive": "pro",
    "company_simulation": "pro",
    "hiring_manager": "ultra", "final_round": "ultra", "technical_deep_dive": "ultra", "cv_attack": "ultra",
    "stress": "ultra", "grill": "ultra", "weakness_targeting": "ultra",
}
PLAN_RANK = {"free": 0, "pro": 1, "ultra": 2}
PLAN_LABEL = {"free": "Free", "pro": "Pro", "ultra": "Ultra"}


def plan_level(f: dict, decision) -> str:
    """The plan whose limits apply: free | pro | ultra. Test grants, admins (and Pro while
    `plans.pro_limits` is off) are not limited: "ultra" level."""
    via = getattr(decision, "via", None)
    if via == TRIAL:
        return "free"
    if via == PRO and f.get("plans.pro_limits", True):
        return "pro"
    return "ultra"


def required_plan(f: dict, decision, mode: str) -> Optional[str]:
    """None if this account may use `mode`, else the plan that includes it."""
    need = MODE_PLAN.get(mode, "ultra")
    return None if PLAN_RANK[plan_level(f, decision)] >= PLAN_RANK[need] else need


def locked_modes(f: dict, decision) -> dict:
    """{mode: plan that includes it} for the modes this account cannot use (empty = all)."""
    return {m: need for m in MODE_PLAN if (need := required_plan(f, decision, m))}


def durations_for(f: dict, decision) -> list:
    """The interview lengths this account may choose. Free and Pro have one length each."""
    level = plan_level(f, decision)
    if level == "free":
        return [int(f["plans.trial_minutes"])]
    if level == "pro":
        return [int(f["plans.pro_interview_minutes"])]
    return list(f["limits.allowed_durations"])


def mode_plans() -> list:
    """For the plans page: every interview type with the plan that first includes it."""
    from ..interview_engine.modes import MODES
    return [{"id": m, "label": MODES[m].label, "plan": need} for m, need in MODE_PLAN.items() if m in MODES]


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
    """The voice engine this account's calls use. Free and Pro interviews can run on a cheaper
    engine (`plans.trial_voice_engine`, `plans.pro_voice_engine`) than Ultra; "same" = the global
    `voice.engine`."""
    return engines_by_plan(flags.all_flags(db))[plan_level(flags.all_flags(db), decision)]


def engines_by_plan(f: dict) -> dict:
    """{free, pro, ultra: voice engine} — also shown on the plans page, so a difference is said."""
    main = f.get("voice.engine", "realtime")
    trial, pro = f.get("plans.trial_voice_engine", "same"), f.get("plans.pro_voice_engine", "same")
    return {"free": main if trial == "same" else trial, "pro": main if pro == "same" else pro, "ultra": main}


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
        "pro_limits": bool(f["plans.pro_limits"]),
        "pro_interview_minutes": int(f["plans.pro_interview_minutes"]),
        "pro_monthly_interviews": int(f["plans.pro_monthly_interviews"]),
        "pro_voice_engine": f["plans.pro_voice_engine"],
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
    f = flags.all_flags(db)
    out: dict = {"via": decision.via, "level": plan_level(f, decision) if decision.allowed else None,
                 "visible": visible(db, decision, user_id), "trial": None, "pro": None, "ultra": None,
                 "locked_modes": locked_modes(f, decision) if decision.allowed else {}}
    if decision.via == TRIAL or decision.via_hint == TRIAL:
        st = trial_state(db, user_id)
        out["trial"] = {"minutes": cfg["trial_minutes"], "available": not st["started"],
                        "in_progress": st["in_progress"], "used": st["started"] and not st["in_progress"],
                        "session_id": st["session_id"]}
    if decision.via == PRO and cfg["pro_limits"]:
        used = started_last_30d(db, user_id)
        out["pro"] = {"minutes": cfg["pro_interview_minutes"], "monthly_interviews": cfg["pro_monthly_interviews"],
                      "started_last_30d": used, "left": max(0, cfg["pro_monthly_interviews"] - used)}
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
        "pro": {"minutes": cfg["pro_interview_minutes"], "monthly_interviews": cfg["pro_monthly_interviews"],
                "open": bool(flags.all_flags(db)["ii.enabled_for_pro"])},
        "voice": engines_by_plan(flags.all_flags(db)),
        "ultra": {"price_inr": cfg["ultra_price_inr"], "monthly_interviews": cfg["ultra_monthly_interviews"],
                  "durations": list(flags.all_flags(db)["limits.allowed_durations"]),
                  "interested": already_interested(db, user_id)},
        "modes": mode_plans(),
        "you": summary(db, decision, user_id),
    }
