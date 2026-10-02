"""Document lifecycle: validate -> hash -> dedupe -> parse -> redact -> encrypt -> store ->
enqueue analysis + Drive sync. The II database holds the durable copy until Drive confirms."""

from __future__ import annotations

import hashlib
import uuid
from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ..access import flags
from ..access.audit import audit
from ..ai.guard import scan_injection
from ..config import get_settings
from ..db.models import Document, DocumentAnalysis, DocumentContent, User, utcnow
from ..errors import NotFound, Unprocessable
from ..jobs import progress
from ..jobs.queue import enqueue
from ..security.crypto import decrypt_text, encrypt_bytes, encrypt_text
from ..versions import ENGINE_VERSIONS
from . import parsing, pii
from .validation import detect, validate_pasted_text

ANALYSIS_VERSION = {"cv": ENGINE_VERSIONS["cv_analysis"], "jd": ENGINE_VERSIONS["jd_analysis"]}


def _next_version(db: Session, user_id: uuid.UUID, kind: str) -> int:
    v = db.execute(select(func.max(Document.version)).where(Document.user_id == user_id, Document.kind == kind)).scalar()
    return int(v or 0) + 1


def _find_duplicate(db: Session, user_id: uuid.UUID, kind: str, sha: str) -> Optional[Document]:
    return db.execute(
        select(Document).where(Document.user_id == user_id, Document.kind == kind, Document.sha256 == sha,
                               Document.deleted_at.is_(None))
    ).scalars().first()


def _store(db: Session, user: User, *, kind: str, source: str, label: str, file_name: str, mime: str, ext: str,
           raw: Optional[bytes], parsed: parsing.ParseResult, sha: str) -> Document:
    redacted, counts = pii.redact(parsed.text, kind)
    flags_found = scan_injection(parsed.text)
    quality = dict(parsed.quality or {})
    quality["redactions"] = counts
    text_sha = hashlib.sha256(" ".join((parsed.text or "").lower().split()).encode("utf-8")).hexdigest() \
        if parsed.text else ""
    doc = Document(
        user_id=user.id, kind=kind, source=source, label=(label or "")[:200], file_name=(file_name or "")[:255],
        mime_type=mime, ext=ext, size_bytes=len(raw) if raw else len(parsed.text.encode()), sha256=sha,
        text_sha256=text_sha,
        version=_next_version(db, user.id, kind), page_count=parsed.page_count, text_chars=len(parsed.text),
        text_quality=quality, parse_status="parsed" if parsed.ok else "unreadable",
        parse_error="" if parsed.ok else parsed.user_message, injection_flags=flags_found,
        storage_status="pending" if raw else "disabled",
    )
    db.add(doc)
    db.flush()
    db.add(DocumentContent(
        document_id=doc.id,
        blob_enc=encrypt_bytes(raw) if raw else None,
        text_enc=encrypt_text(parsed.text) if parsed.text else None,
        redacted_text_enc=encrypt_text(redacted) if redacted else None,
    ))
    if parsed.ok:
        db.add(DocumentAnalysis(document_id=doc.id, user_id=user.id, kind=kind, sha256=sha,
                                analysis_version=ANALYSIS_VERSION[kind], status="pending"))
        enqueue(db, "analyze_document", {"document_id": str(doc.id)}, dedupe_key=f"analyze:{doc.id}",
                max_attempts=3)
    if raw is not None:
        if get_settings().drive_configured:
            enqueue(db, "drive_sync_document", {"document_id": str(doc.id)}, dedupe_key=f"drive:{doc.id}",
                    max_attempts=6)
        else:
            doc.storage_status = "disabled"
    audit(db, "document.create", actor_user_id=user.id, actor_email=user.email, target_type="document",
          target_id=doc.id, meta={"kind": kind, "source": source, "parsed": parsed.ok, "injection_flags": flags_found})
    return doc


def create_from_upload(db: Session, user: User, *, kind: str, data: bytes, filename: str, label: str = "") -> Document:
    if kind not in ("cv", "jd"):
        raise Unprocessable("kind must be 'cv' or 'jd'.", code="bad_kind")
    max_mb = int(flags.flag(db, "limits.max_upload_mb") or 5)
    detected = detect(data, filename, max_mb * 1024 * 1024)
    sha = hashlib.sha256(data).hexdigest()
    dup = _find_duplicate(db, user.id, kind, sha)
    if dup is not None:
        return dup
    parsed = parsing.parse(data, detected.ext, kind)
    return _store(db, user, kind=kind, source="upload", label=label, file_name=filename, mime=detected.mime,
                  ext=detected.ext, raw=data, parsed=parsed, sha=sha)


def create_from_text(db: Session, user: User, *, kind: str, text: str, label: str = "") -> Document:
    if kind != "jd":
        raise Unprocessable("Only job descriptions can be pasted as text. Upload your CV as a file.", code="bad_kind")
    clean = validate_pasted_text(text)
    sha = hashlib.sha256(clean.encode("utf-8")).hexdigest()
    dup = _find_duplicate(db, user.id, kind, sha)
    if dup is not None:
        return dup
    parsed = parsing.parse_plain(clean, kind)
    return _store(db, user, kind=kind, source="paste", label=label, file_name="", mime="text/plain", ext="txt",
                  raw=clean.encode("utf-8"), parsed=parsed, sha=sha)


def get_owned(db: Session, user_id: uuid.UUID, doc_id: uuid.UUID, *, for_update: bool = False) -> Document:
    q = select(Document).where(Document.id == doc_id, Document.user_id == user_id, Document.deleted_at.is_(None))
    if for_update:
        q = q.with_for_update()
    doc = db.execute(q).scalar_one_or_none()
    if doc is None:
        raise NotFound("Document not found.")
    return doc


def latest_analysis(db: Session, doc: Document) -> Optional[DocumentAnalysis]:
    return db.execute(
        select(DocumentAnalysis).where(DocumentAnalysis.document_id == doc.id,
                                       DocumentAnalysis.analysis_version == ANALYSIS_VERSION[doc.kind])
    ).scalar_one_or_none()


def redacted_text(db: Session, doc_id: uuid.UUID) -> str:
    c = db.get(DocumentContent, doc_id)
    return decrypt_text(c.redacted_text_enc) if c else ""


def list_owned(db: Session, user_id: uuid.UUID, kind: Optional[str] = None) -> List[Document]:
    q = select(Document).where(Document.user_id == user_id, Document.deleted_at.is_(None))
    if kind:
        q = q.where(Document.kind == kind)
    return list(db.execute(q.order_by(Document.created_at.desc()).limit(50)).scalars())


def delete_owned(db: Session, user: User, doc_id: uuid.UUID) -> None:
    doc = get_owned(db, user.id, doc_id, for_update=True)
    doc.deleted_at = utcnow()
    c = db.get(DocumentContent, doc.id)
    if c is not None:
        c.blob_enc = None
        c.text_enc = None
        c.redacted_text_enc = None
        c.purged_at = utcnow()
    if doc.storage_status in ("synced", "pending", "uploading", "failed"):
        enqueue(db, "drive_delete", {"target_kind": doc.kind, "target_id": str(doc.id)},
                dedupe_key=f"drive_delete:{doc.id}", max_attempts=6)
    audit(db, "document.delete", actor_user_id=user.id, actor_email=user.email, target_type="document",
          target_id=doc.id)


# What the analysis looks for, shown one at a time while it runs (these are the fields the
# CV / JD parser actually extracts — a description of the step, not a fake sub-progress).
LOOKING_FOR = {
    "cv": ["your roles, dates and scope", "achievements and the numbers behind them", "skills and tools",
           "claims an interviewer would want to verify"],
    "jd": ["the role, team and seniority", "must-have and good-to-have requirements",
           "responsibilities and how success is measured", "keywords an interviewer will listen for"],
}
_BASE_PCT = 10  # reading + privacy are done by the time the upload returns


def _findings(kind: str, result: dict) -> tuple:
    """Real numbers from the parsed document, and a few chips (skills / JD keywords)."""
    r = result or {}
    if kind == "cv":
        exp = r.get("experience") or []
        orgs = {(x.get("organization") or "").strip().lower() for x in exp if (x.get("organization") or "").strip()}
        years = r.get("total_experience_years")
        claims = r.get("claims") or []
        quantified = sum(1 for c in claims if c.get("quantified") or c.get("metrics"))
        flagged = set((r.get("claim_flags") or {}).keys())
        probe = sum(1 for c in claims if c.get("verification_priority") == 1 or c.get("id") in flagged)
        sk = r.get("skills") or {}
        skills: list = []
        for group in ("technical", "tools", "domain"):
            for x in sk.get(group) or []:
                if x and x.lower() not in {s.lower() for s in skills}:
                    skills.append(x)
        facts = [
            progress.join([progress.plural(len(exp), "role") if exp else "",
                           progress.plural(len(orgs), "organisation") if orgs else "",
                           f"{round(float(years) * 2) / 2:g} years" if isinstance(years, (int, float)) and years > 0 else ""]),
            (f"{progress.plural(len(claims), 'achievement')}, {quantified} with numbers" if claims else ""),
            (progress.plural(len(skills), "skill or tool", "skills and tools") if skills else ""),
            (f"{progress.plural(probe, 'claim')} the interviewer will want to dig into" if probe else ""),
        ]
        return [f for f in facts if f], [c for c in (progress.clean_chip(x) for x in skills) if c][:6]
    ident = r.get("identity") or {}
    reqs = r.get("requirements") or []
    must = sum(1 for x in reqs if x.get("importance") == "must")
    other = len(reqs) - must
    sen = (r.get("seniority") or {}).get("level") or ""
    resp = r.get("responsibilities") or []
    kws = sorted(r.get("keywords") or [], key=lambda k: {"high": 0, "medium": 1, "low": 2}.get(k.get("weight"), 1))
    facts = [
        progress.join([ident.get("title") or ident.get("role_name") or "", progress.seniority_label(sen)]),
        progress.join([progress.plural(must, "must-have") if must else "",
                       progress.plural(other, "good-to-have") if other else ""]),
        progress.plural(len(resp), "responsibility", "responsibilities") if resp else "",
    ]
    chips: list = []
    for k in kws:
        t = progress.clean_chip(k.get("term"))
        if t and t.lower() not in {c.lower() for c in chips}:
            chips.append(t)
    return [f for f in facts if f], chips[:8]


def analysis_progress(doc: Document, a: Optional[DocumentAnalysis]) -> Optional[dict]:
    """Step-by-step progress of reading and analysing a document, for the page to show while the
    person waits — and what was found, the moment it is ready."""
    if doc.parse_status != "parsed":
        return None
    status = a.status if a else "pending"
    if status in ("failed", "unreadable"):
        return None
    cv = doc.kind == "cv"
    q = doc.text_quality or {}
    words = int(q.get("words") or 0)
    steps = [{"id": "read", "label": "Read your CV" if cv else "Read the job description", "state": "done",
              "detail": "", "facts": [progress.join([progress.plural(doc.page_count, "page") if doc.page_count else "",
                                                      f"{words:,} words" if words else ""])]}]
    if cv:
        red = q.get("redactions") or {}
        contact = sum(int(red.get(k) or 0) for k in ("emails", "phones", "urls"))
        personal = sum(int(red.get(k) or 0) for k in ("protected_lines", "id_numbers"))
        facts = []
        if contact:
            facts.append(f"{progress.plural(contact, 'contact detail')} hidden")
        if personal:
            facts.append(f"{progress.plural(personal, 'personal detail')} removed (age, address, IDs…)")
        steps.append({"id": "private", "label": "Kept your personal details away from the AI", "state": "done",
                      "detail": "", "facts": facts or ["Nothing personal to remove"]})
    analyse = {"id": "analyse", "label": "Understanding your experience" if cv else "Understanding the role",
               "state": "todo", "detail": "", "facts": [], "looking_for": LOOKING_FOR[doc.kind]}
    steps.append(analyse)
    out = {"steps": steps, "pct": _BASE_PCT, "done": False, "step_elapsed_s": 0.0, "step_eta_s": 0.0,
           "step_weight": 0.0, "highlights": []}
    if status == "ready":
        facts, chips = _findings(doc.kind, a.result if a else {})
        analyse.update(state="done", facts=facts)
        out.update(pct=100, done=True, highlights=chips)
        return out
    live = progress.view(f"doc:{doc.id}")
    if live and not live["failed"]:
        span = 100 - _BASE_PCT
        analyse["state"] = "active"
        analyse["detail"] = "Saving what we found…" if live["done"] else ""
        out.update(pct=min(99, _BASE_PCT + int(live["pct"] * span / 100)),
                   step_elapsed_s=live["step_elapsed_s"], step_eta_s=live["step_eta_s"],
                   step_weight=round(span / 100 * live["step_weight"], 4))
    else:
        analyse["state"] = "waiting"
        analyse["detail"] = "Starting in a moment…"
    return out


def public_view(db: Session, doc: Document, *, include_analysis: bool = False, include_progress: bool = True) -> dict:
    a = latest_analysis(db, doc)
    out = {
        "id": str(doc.id), "kind": doc.kind, "source": doc.source, "label": doc.label,
        "file_name": doc.file_name, "ext": doc.ext, "size_bytes": doc.size_bytes, "version": doc.version,
        "page_count": doc.page_count, "parse_status": doc.parse_status, "parse_error": doc.parse_error or None,
        "warnings": (doc.text_quality or {}).get("warnings", []),
        "storage_status": doc.storage_status, "created_at": doc.created_at.isoformat(),
        "analysis_status": a.status if a else ("unreadable" if doc.parse_status != "parsed" else "pending"),
    }
    if include_analysis and a is not None and a.status == "ready":
        out["analysis"] = a.result
        out["analysis_quality"] = a.quality
    if include_progress:
        out["progress"] = analysis_progress(doc, a)
    return out
