"""Candidate-facing API (docs/C_API_CONTRACT.md §2)."""

from __future__ import annotations

import uuid
from typing import List, Optional

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select

from ..access import flags
from ..auth.assertion import Principal
from ..db.models import InterviewBlueprint, InterviewExchange, InterviewMessage, InterviewState, Report
from ..documents import service as docs
from ..errors import Conflict, NotFound, Unprocessable
from ..interview_engine import orchestrator, sessions
from ..interview_engine.lifecycle import active_sessions, find_owned, sessions_today
from ..report_engine import history
from .deps import principal, read_own, unit, use

router = APIRouter(prefix="/v1")


def _uid(raw: str) -> uuid.UUID:
    try:
        return uuid.UUID(raw)
    except ValueError:
        raise NotFound("Not found.")


# ------------------------------------------------------------------ identity ----------
@router.get("/access")
def access(p: Principal = Depends(principal)):
    """Cheap check for the MECE app nav: may this user open Interview Intelligence right now?
    The nav calls it once per page load, so it does one grant lookup at most (flags are
    cached) and never registers the user. Same decision as every other route."""
    from ..access import policy, rate_limit
    rate_limit.check(str(p.user_id), "read")
    with unit() as db:
        d = policy.decide(db, p)
        return {"allowed": d.allowed, "via": d.via, "is_admin": d.is_admin}


@router.get("/me")
def me(p: Principal = Depends(principal)):
    with unit() as db:
        user, d = read_own(db, p)
        f = flags.all_flags(db)
        act = active_sessions(db, user.id)
        return {
            "user": {"id": str(user.id), "email": user.email},
            "access": {"allowed": d.allowed, "via": d.via, "reason": d.reason},
            "is_admin": d.is_admin,
            "limits": {"max_active_sessions": f["limits.max_active_sessions"], "active_count": len(act),
                       "active_sessions": [{"id": str(s.id), "status": s.status, "role_title": s.role_title}
                                           for s in act],
                       "sessions_last_24h": sessions_today(db, user.id),
                       "max_sessions_per_day": f["limits.max_sessions_per_day_test"] if d.via in ("test_grant", "admin")
                       else f["limits.max_sessions_per_day"],
                       "allowed_durations": f["limits.allowed_durations"], "max_upload_mb": f["limits.max_upload_mb"]},
            "flags": {"voice": f["voice.enabled"], "voice_engine": f.get("voice.engine", "realtime"),
                      "company_intel": f["company_intel.enabled"],
                      "advanced_technical": f["technical.advanced_mode"]},
        }


# ------------------------------------------------------------------ documents ---------
@router.post("/documents", status_code=201)
def upload_document(kind: str = Form(...), label: str = Form(""), file: UploadFile = File(...),
                    p: Principal = Depends(principal)):
    # Sync handler (threadpool): parsing + DB work must not block the event loop.
    data = file.file.read(25 * 1024 * 1024 + 1)  # hard ceiling; the configured cap is applied in validation
    with unit() as db:
        user, _ = use(db, p, klass="upload")
        doc = docs.create_from_upload(db, user, kind=kind, data=data, filename=file.filename or "", label=label)
        return docs.public_view(db, doc)


class PasteBody(BaseModel):
    kind: str = "jd"
    text: str
    label: str = ""


@router.post("/documents/text", status_code=201)
def paste_document(body: PasteBody, p: Principal = Depends(principal)):
    with unit() as db:
        user, _ = use(db, p, klass="upload")
        doc = docs.create_from_text(db, user, kind=body.kind, text=body.text, label=body.label)
        return docs.public_view(db, doc)


@router.get("/documents")
def list_documents(kind: Optional[str] = Query(None), p: Principal = Depends(principal)):
    with unit() as db:
        user, _ = read_own(db, p)
        return {"documents": [docs.public_view(db, d) for d in docs.list_owned(db, user.id, kind)]}


@router.get("/documents/{doc_id}")
def get_document(doc_id: str, p: Principal = Depends(principal)):
    with unit() as db:
        user, _ = read_own(db, p)
        return docs.public_view(db, docs.get_owned(db, user.id, _uid(doc_id)), include_analysis=True)


@router.delete("/documents/{doc_id}", status_code=204)
def delete_document(doc_id: str, p: Principal = Depends(principal)):
    with unit() as db:
        user, _ = read_own(db, p)
        docs.delete_owned(db, user, _uid(doc_id))
    return Response(status_code=204)


# ------------------------------------------------------------------ sessions ----------
class SessionBody(BaseModel):
    cv_document_id: str
    jd_document_id: str
    config: dict = Field(default_factory=dict)
    source_session_id: Optional[str] = None


@router.post("/sessions", status_code=201)
def create_session(body: SessionBody, p: Principal = Depends(principal)):
    with unit() as db:
        user, d = use(db, p, klass="session_create")
        s = sessions.create_session(db, user, d, cv_document_id=_uid(body.cv_document_id),
                                    jd_document_id=_uid(body.jd_document_id), config=body.config,
                                    source_session_id=_uid(body.source_session_id) if body.source_session_id else None)
        return sessions.view(db, s)


@router.get("/sessions")
def list_sessions(p: Principal = Depends(principal)):
    with unit() as db:
        user, _ = read_own(db, p)
        return {"sessions": [sessions.view(db, s, detail=False) for s in sessions.list_for(db, user.id)]}


@router.get("/sessions/{sid}")
def get_session(sid: str, p: Principal = Depends(principal)):
    with unit() as db:
        user, _ = read_own(db, p)
        s = find_owned(db, user.id, _uid(sid))
        if s is None:
            raise NotFound("Interview not found.")
        out = sessions.view(db, s)
        st = db.execute(select(InterviewState).where(InterviewState.session_id == s.id)).scalar_one_or_none()
        bp = db.execute(select(InterviewBlueprint).where(InterviewBlueprint.session_id == s.id)).scalar_one_or_none()
        out["progress"] = orchestrator.session_progress(s, st.state if st else None, bp.blueprint if bp else None)
        return out


@router.get("/sessions/{sid}/room")
def room(sid: str, p: Principal = Depends(principal)):
    """Everything the interview room needs to (re)render: transcript so far + progress.
    Contains no evaluation signals (spec §78)."""
    with unit() as db:
        user, _ = read_own(db, p)
        s = find_owned(db, user.id, _uid(sid))
        if s is None:
            raise NotFound("Interview not found.")
        st = db.execute(select(InterviewState).where(InterviewState.session_id == s.id)).scalar_one_or_none()
        bp = db.execute(select(InterviewBlueprint).where(InterviewBlueprint.session_id == s.id)).scalar_one_or_none()
        msgs = db.execute(select(InterviewMessage).where(InterviewMessage.session_id == s.id)
                          .order_by(InterviewMessage.seq)).scalars().all()
        return {"session": sessions.view(db, s, detail=False),
                "progress": orchestrator.session_progress(s, st.state if st else None, bp.blueprint if bp else None),
                "messages": [orchestrator._public_message(m) for m in msgs if m.role != "system"]}


@router.post("/sessions/{sid}/start")
def start(sid: str, p: Principal = Depends(principal)):
    with unit() as db:
        user, _ = use(db, p, klass="turn")
        return orchestrator.start(db, user.id, _uid(sid))


class TurnBody(BaseModel):
    client_turn_id: str
    content: str = ""
    kind: str = "text"
    answer_ms: Optional[int] = None
    skip: bool = False


@router.post("/sessions/{sid}/turns")
def take_turn(sid: str, body: TurnBody, p: Principal = Depends(principal)):
    with unit() as db:
        user, _ = use(db, p, klass="turn")
        return orchestrator.turn(db, user.id, _uid(sid), client_turn_id=body.client_turn_id, content=body.content,
                                 kind=body.kind, answer_ms=body.answer_ms, skip=body.skip)


@router.post("/sessions/{sid}/pause")
def pause(sid: str, p: Principal = Depends(principal)):
    with unit() as db:
        user, _ = use(db, p, klass="turn")
        return orchestrator.pause(db, user.id, _uid(sid))


@router.post("/sessions/{sid}/resume")
def resume(sid: str, p: Principal = Depends(principal)):
    with unit() as db:
        user, _ = use(db, p, klass="turn")
        return orchestrator.resume(db, user.id, _uid(sid))


@router.post("/sessions/{sid}/end")
def end(sid: str, p: Principal = Depends(principal)):
    with unit() as db:
        user, _ = read_own(db, p)  # ending must always be possible, even if access lapsed mid-interview
        return orchestrator.end_early(db, user.id, _uid(sid))


@router.post("/sessions/{sid}/abandon")
def abandon(sid: str, p: Principal = Depends(principal)):
    with unit() as db:
        user, _ = read_own(db, p)
        return orchestrator.abandon(db, user.id, _uid(sid))


@router.get("/sessions/{sid}/transcript")
def transcript(sid: str, p: Principal = Depends(principal)):
    with unit() as db:
        user, _ = read_own(db, p)
        s = find_owned(db, user.id, _uid(sid))
        if s is None:
            raise NotFound("Interview not found.")
        if s.status in ("active", "paused", "ready"):
            raise Conflict("The transcript is available after the interview.", code="in_progress")
        msgs = db.execute(select(InterviewMessage).where(InterviewMessage.session_id == s.id)
                          .order_by(InterviewMessage.seq)).scalars().all()
        exs = {e.id: e.seq for e in db.execute(select(InterviewExchange).where(InterviewExchange.session_id == s.id)).scalars()}
        return {"messages": [orchestrator._public_message(m) | {"exchange_ref": f"X{exs[m.exchange_id]}"
                                                                 if m.exchange_id in exs else None}
                             for m in msgs if m.role != "system"]}


@router.get("/sessions/{sid}/report")
def report(sid: str, p: Principal = Depends(principal)):
    with unit() as db:
        user, _ = read_own(db, p)
        s = find_owned(db, user.id, _uid(sid))
        if s is None:
            raise NotFound("Interview not found.")
        if s.assessment_status in ("pending", "processing"):
            return JSONResponse(status_code=202, content={"status": s.assessment_status,
                                                          "message": "We're analysing your responses, evidence and role alignment."})
        if s.assessment_status == "failed":
            return {"status": "failed", "message": "We couldn't generate this report. Your transcript is saved; "
                                                   "please contact support or retry later."}
        rep = db.execute(select(Report).where(Report.session_id == s.id)).scalar_one_or_none()
        if rep is None:
            return JSONResponse(status_code=404, content={"error": {"code": "no_report",
                                                                    "message": "No report for this interview."}})
        return {"status": rep.status, "report": rep.report}


@router.get("/sessions/{sid}/questions/{exchange_id}/why")
def why(sid: str, exchange_id: str, p: Principal = Depends(principal)):
    with unit() as db:
        user, _ = read_own(db, p)
        s = find_owned(db, user.id, _uid(sid))
        if s is None:
            raise NotFound("Interview not found.")
        rep = db.execute(select(Report).where(Report.session_id == s.id)).scalar_one_or_none()
        q = next((q for q in (rep.report.get("questions", []) if rep else []) if q["exchange_id"] == exchange_id), None)
        if q is None:
            raise NotFound("Question not found.")
        return {"exchange_id": exchange_id, "question": q["question"], "why_asked": q["why_asked"]}


class ReattemptBody(BaseModel):
    target_competencies: List[str] = Field(default_factory=list)
    duration_minutes: Optional[int] = None
    difficulty: Optional[str] = None


@router.post("/sessions/{sid}/reattempt", status_code=201)
def reattempt(sid: str, body: ReattemptBody, p: Principal = Depends(principal)):
    with unit() as db:
        user, d = use(db, p, klass="session_create")
        src = find_owned(db, user.id, _uid(sid))
        if src is None:
            raise NotFound("Interview not found.")
        if src.status not in ("completed", "expired"):
            raise Conflict("You can re-attempt an interview once it has finished.", code="not_finished")
        cfg = dict(src.config or {})
        cfg.update({"mode": "weakness_targeting", "target_competencies": body.target_competencies,
                    "duration_minutes": body.duration_minutes or min(30, int(cfg.get("duration_minutes", 30))),
                    "difficulty": body.difficulty or cfg.get("difficulty", "medium")})
        if cfg["duration_minutes"] not in flags.flag(db, "limits.allowed_durations"):
            raise Unprocessable("Unsupported duration.", code="bad_duration")
        s = sessions.create_session(db, user, d, cv_document_id=src.cv_document_id, jd_document_id=src.jd_document_id,
                                    config=cfg, source_session_id=src.id)
        return sessions.view(db, s)


@router.get("/progress")
def progress(p: Principal = Depends(principal)):
    with unit() as db:
        user, _ = read_own(db, p)
        return history.progress(db, user.id)
