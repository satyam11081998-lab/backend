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


def public_view(db: Session, doc: Document, *, include_analysis: bool = False) -> dict:
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
    return out
