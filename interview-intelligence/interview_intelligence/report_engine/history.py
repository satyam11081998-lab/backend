"""Progress over time + recurring-weakness engine (spec §43, §44).

Only COMPARABLE dimensions are compared: the same canonical competency, scored (not
'not sufficiently tested') with at least moderate confidence in both interviews."""

from __future__ import annotations

import uuid
from collections import Counter, defaultdict
from typing import Dict, List

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..db.models import CompetencyHistory, FeedbackItem, InterviewSession, Report
from ..role_taxonomy.library import library

COMPARABLE_CONF = {"high", "moderate"}


def write_history(db: Session, sess: InterviewSession, assessments: List[dict]) -> None:
    db.query(CompetencyHistory).filter(CompetencyHistory.session_id == sess.id).delete()
    for a in assessments:
        db.add(CompetencyHistory(user_id=sess.user_id, session_id=sess.id, competency_id=a["competency_id"],
                                 canonical_competency_id=a["canonical_competency_id"], role_family=sess.role_family,
                                 score=a.get("score"), evidence_state=a["evidence_state"], confidence=a["confidence"]))


def _assessed_sessions(db: Session, user_id: uuid.UUID, limit: int = 12) -> List[InterviewSession]:
    return list(db.execute(
        select(InterviewSession).join(Report, Report.session_id == InterviewSession.id)
        .where(InterviewSession.user_id == user_id, InterviewSession.deleted_at.is_(None))
        .order_by(InterviewSession.created_at.desc()).limit(limit)).scalars())


def progress(db: Session, user_id: uuid.UUID, *, current_session_id: uuid.UUID | None = None) -> dict:
    lib = library()
    rows = db.execute(select(CompetencyHistory, InterviewSession.created_at)
                      .join(InterviewSession, InterviewSession.id == CompetencyHistory.session_id)
                      .where(CompetencyHistory.user_id == user_id, InterviewSession.deleted_at.is_(None))
                      .order_by(InterviewSession.created_at)).all()
    series: Dict[str, List[dict]] = defaultdict(list)
    for h, created in rows:
        series[h.canonical_competency_id].append({
            "session_id": str(h.session_id), "date": created.isoformat(), "score": h.score,
            "state": h.evidence_state, "confidence": h.confidence, "role_family": h.role_family,
            "comparable": h.score is not None and h.confidence in COMPARABLE_CONF,
        })
    trajectories = []
    for cid, pts in series.items():
        comp = lib.competency(cid)
        comparable = [p for p in pts if p["comparable"]]
        delta = None
        if len(comparable) >= 2:
            delta = comparable[-1]["score"] - comparable[-2]["score"]
        trajectories.append({"competency_id": cid, "name": comp.name if comp else cid, "points": pts,
                             "comparable_points": len(comparable), "latest_delta": delta})
    trajectories.sort(key=lambda t: (-t["comparable_points"], t["name"]))
    deltas = []
    if current_session_id:
        for t in trajectories:
            comp_pts = [p for p in t["points"] if p["comparable"]]
            cur = next((p for p in reversed(comp_pts) if p["session_id"] == str(current_session_id)), None)
            prev = [p for p in comp_pts if p["session_id"] != str(current_session_id) and p["date"] < (cur or {}).get("date", "")]
            if cur and prev:
                deltas.append({"competency_id": t["competency_id"], "name": t["name"], "previous": prev[-1]["score"],
                               "current": cur["score"], "delta": cur["score"] - prev[-1]["score"],
                               "previous_session_id": prev[-1]["session_id"]})
    return {"trajectories": trajectories, "deltas": deltas,
            "recurring": recurring_patterns(db, user_id)}


def recurring_patterns(db: Session, user_id: uuid.UUID) -> List[dict]:
    sessions = _assessed_sessions(db, user_id, limit=3)
    if len(sessions) < 2:
        return []
    ids = [s.id for s in sessions]
    items = db.execute(select(FeedbackItem).where(FeedbackItem.session_id.in_(ids), FeedbackItem.kind == "development")
                       ).scalars().all()
    by_cat: Dict[str, set] = defaultdict(set)
    titles: Dict[str, List[str]] = defaultdict(list)
    for it in items:
        by_cat[it.category].add(str(it.session_id))
        titles[it.category].append(it.title)
    out = []
    for cat, sids in by_cat.items():
        if len(sids) >= 2:
            out.append({"category": cat, "label": cat.replace("_", " "), "occurrences": len(sids),
                        "of_last": len(sessions), "session_ids": sorted(sids),
                        "examples": titles[cat][:3]})
    # weak canonical competencies recurring in history
    hist = db.execute(select(CompetencyHistory).where(CompetencyHistory.session_id.in_(ids))).scalars().all()
    weak = Counter(h.canonical_competency_id for h in hist if h.evidence_state == "weak")
    lib = library()
    for cid, n in weak.items():
        if n >= 2:
            c = lib.competency(cid)
            out.append({"category": f"competency:{cid}", "label": c.name if c else cid, "occurrences": n,
                        "of_last": len(sessions), "session_ids": [], "examples": []})
    out.sort(key=lambda r: -r["occurrences"])
    return out
