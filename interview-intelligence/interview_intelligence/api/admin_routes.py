"""Admin control center (docs/J_ADMIN_ACCESS.md). Every route requires is_ii_admin and
every mutation (and every session inspection) is audit-logged."""

from __future__ import annotations

import uuid
from datetime import timedelta
from statistics import median
from typing import Any, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response
from pydantic import BaseModel
from sqlalchemy import func, select

from ..access import flags
from ..access.audit import audit
from ..auth.assertion import Principal
from ..db.models import (AccessGrant, AuditLog, CompetencyAssessment, Document, DriveFile, EvaluationRun,
                         InterviewEvent, InterviewExchange, InterviewMessage, InterviewSession, ModelRun, User, utcnow)
from ..drive_integration.sync import retry_failed
from ..errors import NotFound, Unprocessable
from ..interview_engine.lifecycle import SLOT_STATES
from ..role_taxonomy.library import library
from ..versions import versions
from ..ai.prompts import prompt_versions
from .deps import admin, principal, unit

router = APIRouter(prefix="/v1/admin")


def _grant_view(g: AccessGrant) -> dict:
    return {"id": str(g.id), "email": g.email_lc, "status": g.status, "grant_type": g.grant_type, "note": g.note,
            "granted_by": g.granted_by, "created_at": g.created_at.isoformat(), "updated_at": g.updated_at.isoformat(),
            "expires_at": g.expires_at.isoformat() if g.expires_at else None,
            "active": g.status == "enabled" and (g.expires_at is None or _aware(g.expires_at) > utcnow())}


def _id(raw: str) -> uuid.UUID:
    try:
        return uuid.UUID(raw)
    except ValueError:
        raise NotFound("Not found.")


def _aware(dt):
    return dt if dt.tzinfo else dt.replace(tzinfo=utcnow().tzinfo)


@router.get("/access-grants")
def list_grants(p: Principal = Depends(principal)):
    with unit() as db:
        admin(db, p)
        rows = db.execute(select(AccessGrant).order_by(AccessGrant.created_at.desc())).scalars().all()
        return {"grants": [_grant_view(g) for g in rows]}


class GrantBody(BaseModel):
    email: str
    note: str = ""


@router.post("/access-grants", status_code=201)
def add_grant(body: GrantBody, p: Principal = Depends(principal)):
    email = (body.email or "").strip().lower()
    if "@" not in email or "." not in email.split("@")[-1] or len(email) > 320 or " " in email:
        raise Unprocessable("Enter a valid email address.", code="bad_email")
    with unit() as db:
        actor = admin(db, p)
        g = db.execute(select(AccessGrant).where(AccessGrant.email_lc == email)).scalar_one_or_none()
        if g is None:
            g = AccessGrant(email_lc=email, status="enabled", grant_type="test", note=body.note[:500],
                            granted_by=actor.email)
            db.add(g)
        else:
            g.status, g.updated_at = "enabled", utcnow()
            if body.note:
                g.note = body.note[:500]
        db.flush()
        flags.invalidate()  # access changed: drop caches derived from it (voice gate)
        audit(db, "access_grant.add", actor_user_id=actor.id, actor_email=actor.email, target_type="access_grant",
              target_id=email)
        return _grant_view(g)


class GrantPatch(BaseModel):
    status: Optional[str] = None
    note: Optional[str] = None


@router.patch("/access-grants/{gid}")
def patch_grant(gid: str, body: GrantPatch, p: Principal = Depends(principal)):
    with unit() as db:
        actor = admin(db, p)
        g = db.get(AccessGrant, _id(gid))
        if g is None:
            raise NotFound("Grant not found.")
        if body.status is not None:
            if body.status not in ("enabled", "disabled"):
                raise Unprocessable("status must be enabled or disabled", code="bad_status")
            g.status = body.status
        if body.note is not None:
            g.note = body.note[:500]
        g.updated_at = utcnow()
        flags.invalidate()  # access changed: drop caches derived from it (voice gate)
        audit(db, f"access_grant.{body.status or 'edit'}", actor_user_id=actor.id, actor_email=actor.email,
              target_type="access_grant", target_id=g.email_lc)
        return _grant_view(g)


@router.delete("/access-grants/{gid}", status_code=204)
def delete_grant(gid: str, p: Principal = Depends(principal)):
    with unit() as db:
        actor = admin(db, p)
        g = db.get(AccessGrant, _id(gid))
        if g is None:
            raise NotFound("Grant not found.")
        flags.invalidate()  # access changed: drop caches derived from it (voice gate)
        audit(db, "access_grant.delete", actor_user_id=actor.id, actor_email=actor.email, target_type="access_grant",
              target_id=g.email_lc)
        db.delete(g)
    return Response(status_code=204)


@router.get("/config")
def get_config(p: Principal = Depends(principal)):
    with unit() as db:
        admin(db, p)
        return {"flags": flags.all_flags(db), "defaults": flags.defaults()}


class ConfigPatch(BaseModel):
    values: dict


@router.patch("/config")
def patch_config(body: ConfigPatch, p: Principal = Depends(principal)):
    with unit() as db:
        actor = admin(db, p)
        out = {}
        for k, v in (body.values or {}).items():
            try:
                out[k] = flags.set_flag(db, k, v, actor=actor.email)
            except KeyError:
                raise Unprocessable(f"Unknown setting '{k}'.", code="unknown_flag")
            except (ValueError, TypeError) as e:
                raise Unprocessable(f"Invalid value for '{k}': {e}", code="bad_flag_value")
        audit(db, "config.update", actor_user_id=actor.id, actor_email=actor.email, target_type="config",
              meta={"values": out})
        return {"flags": flags.all_flags(db)}


def _pct(vals, q):
    if not vals:
        return None
    vals = sorted(vals)
    return vals[min(len(vals) - 1, int(q * len(vals)))]


@router.get("/overview")
def overview(p: Principal = Depends(principal)):
    with unit() as db:
        admin(db, p)
        now = utcnow()
        week = now - timedelta(days=7)
        day = now - timedelta(days=1)
        count = lambda q: int(db.execute(q).scalar() or 0)  # noqa: E731
        sessions_7d = db.execute(select(InterviewSession.status, func.count()).where(InterviewSession.created_at >= week)
                                 .group_by(InterviewSession.status)).all()
        by_status = {s: n for s, n in sessions_7d}
        started = sum(n for s, n in by_status.items() if s not in ("created", "uploading", "analyzing", "ready"))
        runs = db.execute(select(ModelRun.stage, ModelRun.latency_ms, ModelRun.status, ModelRun.cost_usd, ModelRun.model)
                          .where(ModelRun.created_at >= day)).all()
        stages = {}
        for st, lat, status, cost, model in runs:
            d = stages.setdefault(st, {"latencies": [], "errors": 0, "runs": 0, "cost": 0.0, "models": set()})
            d["latencies"].append(lat)
            d["runs"] += 1
            d["cost"] += float(cost or 0)
            d["models"].add(model)
            if status not in ("ok", "repaired"):
                d["errors"] += 1
        mode_usage = db.execute(select(InterviewSession.mode, func.count()).where(InterviewSession.created_at >= week)
                                .group_by(InterviewSession.mode)).all()
        q_usage = db.execute(select(InterviewExchange.planned_qid, InterviewExchange.question_text, func.count())
                             .where(InterviewExchange.started_at >= week)
                             .group_by(InterviewExchange.planned_qid, InterviewExchange.question_text)
                             .order_by(func.count().desc()).limit(10)).all()
        anomalies = db.execute(select(CompetencyAssessment.session_id, CompetencyAssessment.competency_id,
                                      CompetencyAssessment.anomaly_flags)
                               .where(CompetencyAssessment.created_at >= week)).all()
        anomaly_rows = [{"session_id": str(s), "competency_id": c, "flags": f} for s, c, f in anomalies
                        if f and any(x not in ("no_evidence_no_model_call", "insufficient_evidence_gate") for x in f)][:30]
        drive = dict(db.execute(select(DriveFile.storage_status, func.count()).group_by(DriveFile.storage_status)).all())
        doc_storage = dict(db.execute(select(Document.storage_status, func.count()).group_by(Document.storage_status)).all())
        return {
            "users_total": count(select(func.count()).select_from(User)),
            "users_active_7d": count(select(func.count()).select_from(User).where(User.last_seen_at >= week)),
            "active_interviews": count(select(func.count()).select_from(InterviewSession)
                                       .where(InterviewSession.status.in_(["active", "paused"]))),
            "slot_sessions": count(select(func.count()).select_from(InterviewSession)
                                   .where(InterviewSession.status.in_(list(SLOT_STATES)))),
            "completed_7d": by_status.get("completed", 0),
            "sessions_7d_by_status": by_status,
            "failure_rate_7d": round(by_status.get("failed", 0) / max(1, started + by_status.get("failed", 0)), 3),
            "ai_stages_24h": {k: {"runs": v["runs"], "errors": v["errors"], "p50_ms": _pct(v["latencies"], 0.5),
                                  "p95_ms": _pct(v["latencies"], 0.95), "cost_usd": round(v["cost"], 4),
                                  "models": sorted(v["models"])} for k, v in stages.items()},
            "cost_today_usd": round(sum(v["cost"] for v in stages.values()), 4),
            "mode_usage_7d": {m: n for m, n in mode_usage},
            "top_questions_7d": [{"qid": q, "text": t[:140], "count": n} for q, t, n in q_usage],
            "evaluation_anomalies_7d": anomaly_rows,
            "drive_sync": {"files": drive, "documents": doc_storage},
            "versions": versions(), "prompt_versions": prompt_versions(),
        }


@router.get("/sessions")
def admin_sessions(status: Optional[str] = Query(None), limit: int = Query(50, le=200), p: Principal = Depends(principal)):
    with unit() as db:
        admin(db, p)
        q = select(InterviewSession, User.email).join(User, User.id == InterviewSession.user_id)
        if status:
            q = q.where(InterviewSession.status == status)
        rows = db.execute(q.order_by(InterviewSession.created_at.desc()).limit(limit)).all()
        return {"sessions": [{"id": str(s.id), "email": e, "status": s.status, "mode": s.mode,
                              "difficulty": s.difficulty, "role_title": s.role_title, "role_family": s.role_family,
                              "created_at": s.created_at.isoformat(), "assessment_status": s.assessment_status,
                              "cost_usd": float(s.cost_usd or 0), "active_seconds": s.active_seconds}
                             for s, e in rows]}


@router.get("/sessions/{sid}")
def admin_session(sid: str, p: Principal = Depends(principal)):
    with unit() as db:
        actor = admin(db, p)
        s = db.get(InterviewSession, _id(sid))
        if s is None:
            raise NotFound("Session not found.")
        audit(db, "admin.inspect_session", actor_user_id=actor.id, actor_email=actor.email, target_type="session",
              target_id=s.id)
        msgs = db.execute(select(InterviewMessage).where(InterviewMessage.session_id == s.id)
                          .order_by(InterviewMessage.seq)).scalars().all()
        events = db.execute(select(InterviewEvent).where(InterviewEvent.session_id == s.id)
                            .order_by(InterviewEvent.created_at)).scalars().all()
        runs = db.execute(select(ModelRun).where(ModelRun.session_id == s.id).order_by(ModelRun.created_at)).scalars().all()
        return {
            "session": {"id": str(s.id), "status": s.status, "config": s.config, "versions": s.versions,
                        "cost_usd": float(s.cost_usd or 0), "ended_reason": s.ended_reason},
            "messages": [{"seq": m.seq, "role": m.role, "content": m.content, "action": m.action, "meta": m.meta}
                         for m in msgs],
            "events": [{"type": e.type, "payload": e.payload, "at": e.created_at.isoformat()} for e in events],
            "model_runs": [{"stage": r.stage, "model": r.model, "prompt": f"{r.prompt_id}@{r.prompt_version}",
                            "status": r.status, "latency_ms": r.latency_ms, "tokens": [r.input_tokens, r.output_tokens],
                            "cost_usd": float(r.cost_usd or 0), "error": r.error[:300]} for r in runs],
        }


@router.get("/model-runs")
def model_runs(stage: Optional[str] = Query(None), status: Optional[str] = Query(None), limit: int = Query(100, le=500),
               p: Principal = Depends(principal)):
    with unit() as db:
        admin(db, p)
        q = select(ModelRun)
        if stage:
            q = q.where(ModelRun.stage == stage)
        if status:
            q = q.where(ModelRun.status == status)
        rows = db.execute(q.order_by(ModelRun.created_at.desc()).limit(limit)).scalars().all()
        return {"runs": [{"at": r.created_at.isoformat(), "stage": r.stage, "provider": r.provider, "model": r.model,
                          "prompt": f"{r.prompt_id}@{r.prompt_version}", "status": r.status, "attempt": r.attempt,
                          "latency_ms": r.latency_ms, "cost_usd": float(r.cost_usd or 0), "error": r.error[:300]}
                         for r in rows]}


@router.get("/library")
def lib_view(p: Principal = Depends(principal)):
    with unit() as db:
        admin(db, p)
        lib = library()
        return {"versions": lib.versions,
                "families": [{"id": f.id, "name": f.name, "technical": f.technical, "defaults": f.defaults}
                             for f in lib.families.values()],
                "competencies": [{"id": c.id, "name": c.name, "category": c.category, "parent": c.parent}
                                 for c in lib.competencies.values()],
                "archetypes": [{"id": a.id, "name": a.name, "type": a.question_type} for a in lib.archetypes.values()],
                "curated_questions": len(lib.bank)}


@router.post("/drive/retry-failed")
def drive_retry(p: Principal = Depends(principal)):
    with unit() as db:
        actor = admin(db, p)
        n = retry_failed(db)
        audit(db, "drive.retry_failed", actor_user_id=actor.id, actor_email=actor.email, meta={"requeued": n})
        return {"requeued": n}


@router.get("/evaluation-runs")
def eval_runs(p: Principal = Depends(principal)):
    with unit() as db:
        admin(db, p)
        rows = db.execute(select(EvaluationRun).order_by(EvaluationRun.created_at.desc()).limit(20)).scalars().all()
        return {"runs": [{"id": str(r.id), "at": r.created_at.isoformat(), "kind": r.kind,
                          "evaluator_version": r.evaluator_version, "dataset_version": r.dataset_version,
                          "metrics": r.metrics, "notes": r.notes} for r in rows]}


@router.get("/audit")
def audit_log(limit: int = Query(100, le=500), p: Principal = Depends(principal)):
    with unit() as db:
        admin(db, p)
        rows = db.execute(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(limit)).scalars().all()
        return {"entries": [{"at": r.created_at.isoformat(), "actor": r.actor_email, "action": r.action,
                             "target": f"{r.target_type}:{r.target_id}", "meta": r.meta} for r in rows]}
