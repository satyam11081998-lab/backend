"""Drive sync jobs: folder ensure (no duplicates), idempotent upload, report export, delete.

The II database keeps the encrypted original until Drive confirms (spec §91: a Drive
failure never loses a document)."""

from __future__ import annotations

import hashlib
import hmac
import uuid
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..access.audit import audit
from ..config import get_settings
from ..db.models import (Document, DocumentContent, DriveFile, DriveFolder, InterviewSession, Report, User, utcnow)
from ..jobs.queue import register
from ..security.crypto import decrypt_bytes
from . import client as drive

ROOT_NAME = "Users"
SUB = {"cv": "CV", "jd": "JD", "report": "Interviews"}


def folder_key(user_id: uuid.UUID) -> str:
    """Opaque, stable, non-reversible folder name: no names or emails in Drive."""
    salt = (get_settings().drive_folder_salt or "ii-default-salt").encode()
    return "u_" + hmac.new(salt, str(user_id).encode(), hashlib.sha256).hexdigest()[:16]


def _ensure(db: Session, scope_key: str, kind: str, name: str, parent: str) -> str:
    row = db.execute(select(DriveFolder).where(DriveFolder.scope_key == scope_key, DriveFolder.kind == kind)).scalar_one_or_none()
    if row is not None:
        return row.folder_id
    # Callers hold the user's row lock (see _locked_user), so two workers cannot both reach
    # this point for the same user; the lookup-before-create covers a crash between the Drive
    # call and our commit.
    fid = drive.find_folder(name, parent) or drive.create_folder(name, parent)
    db.add(DriveFolder(scope_key=scope_key, kind=kind, folder_id=fid, parent_folder_id=parent, name=name))
    db.flush()
    return fid


def _locked_user(db: Session, user_id: uuid.UUID) -> User:
    return db.execute(select(User).where(User.id == user_id).with_for_update()).scalar_one()


def ensure_user_folder(db: Session, user: User, target_kind: str) -> str:
    root = get_settings().gdrive_root_folder_id
    users_root = _ensure(db, "global", "users_root", ROOT_NAME, root)
    if not user.drive_folder_key:
        user.drive_folder_key = folder_key(user.id)
    user_folder = _ensure(db, str(user.id), "user", user.drive_folder_key, users_root)
    return _ensure(db, str(user.id), target_kind, SUB[target_kind], user_folder)


def _drive_row(db: Session, user_id: uuid.UUID, target_kind: str, target_id: str) -> DriveFile:
    row = db.execute(select(DriveFile).where(DriveFile.target_kind == target_kind, DriveFile.target_id == target_id)
                     .with_for_update()).scalar_one_or_none()
    if row is None:
        row = DriveFile(user_id=user_id, target_kind=target_kind, target_id=target_id)
        db.add(row)
        db.flush()
    return row


def _sync_bytes(db: Session, user: User, *, target_kind: str, target_id: str, name: str, data: bytes, mime: str,
                sha: str) -> DriveFile:
    row = _drive_row(db, user.id, target_kind, target_id)
    if row.storage_status == "synced" and row.drive_file_id:
        return row
    row.attempts += 1
    row.storage_status = "uploading"
    row.updated_at = utcnow()
    try:
        folder = ensure_user_folder(db, user, target_kind)
        existing = drive.find_by_app_property("ii_target_id", target_id, folder)  # crash-safe idempotency
        fid = existing["id"] if existing else drive.upload(name, data, mime, folder,
                                                            {"ii_target_id": target_id, "ii_sha256": sha[:64]})
    except drive.DriveError as e:
        row.storage_status = "failed"
        row.last_error = str(e)[:2000]
        raise
    row.drive_file_id, row.drive_folder_id = fid, folder
    row.file_name, row.mime_type, row.sha256 = name, mime, sha
    row.storage_status, row.uploaded_at, row.last_error = "synced", utcnow(), ""
    audit(db, "drive.synced", actor_email="system", target_type=target_kind, target_id=target_id)
    return row


def _doc_failed(db: Session, payload: dict, err: str) -> None:
    doc = db.get(Document, uuid.UUID(payload["document_id"]))
    if doc is not None and doc.storage_status != "synced":
        doc.storage_status = "failed"
    row = db.execute(select(DriveFile).where(DriveFile.target_id == payload["document_id"])).scalar_one_or_none()
    if row is not None:
        row.storage_status, row.last_error = "failed", err[:2000]
    audit(db, "drive.failed", actor_email="system", target_type="document", target_id=payload["document_id"],
          meta={"error": err[:300]})


@register("drive_sync_document", on_dead=_doc_failed)
def handle_sync_document(db: Session, payload: dict) -> None:
    if not get_settings().drive_configured:
        return
    doc = db.get(Document, uuid.UUID(payload["document_id"]))
    if doc is None or doc.deleted_at is not None:
        return
    content = db.get(DocumentContent, doc.id)
    if content is None or content.blob_enc is None:
        doc.storage_status = "failed"
        return
    user = _locked_user(db, doc.user_id)
    data = decrypt_bytes(content.blob_enc)
    ext = doc.ext or "bin"
    name = f"{doc.kind}_{doc.id}_{doc.sha256[:12]}.{ext}"
    doc.storage_status = "uploading"
    try:
        _sync_bytes(db, user, target_kind=doc.kind, target_id=str(doc.id), name=name, data=data,
                    mime=doc.mime_type or "application/octet-stream", sha=doc.sha256)
    except drive.DriveError as e:
        if not e.retryable:
            doc.storage_status = "failed"
            return
        raise
    doc.storage_status = "synced"
    if get_settings().purge_blob_after_drive_sync:
        content.blob_enc = None


def _report_markdown(sess: InterviewSession, rep: dict) -> str:
    h = rep.get("header", {})
    hl = rep.get("headline", {})
    lines = [f"# Interview Intelligence — {h.get('role_title') or 'Interview'}",
             f"Date: {h.get('date', '')} | Mode: {h.get('mode')} | Difficulty: {h.get('difficulty')}",
             f"Role alignment: {hl.get('role_alignment')} | Confidence: {hl.get('assessment_confidence')}", "",
             "## Executive assessment", rep.get("executive_assessment", ""), "", "## Competencies"]
    for c in rep.get("competencies", []):
        lines.append(f"- {c['name']}: {c['band']} (confidence {c['confidence']})")
    lines += ["", "## Development areas"]
    for d in rep.get("development_areas", []):
        lines.append(f"- {d.get('title')}: {d.get('what_to_do')}")
    return "\n".join(lines) + "\n"


@register("drive_export_report")
def handle_export_report(db: Session, payload: dict) -> None:
    if not get_settings().drive_configured:
        return
    sess = db.get(InterviewSession, uuid.UUID(payload["session_id"]))
    rep = db.execute(select(Report).where(Report.session_id == sess.id)).scalar_one_or_none() if sess else None
    if sess is None or rep is None:
        return
    user = _locked_user(db, sess.user_id)
    md = _report_markdown(sess, rep.report).encode("utf-8")
    _sync_bytes(db, user, target_kind="report", target_id=str(sess.id), name=f"report_{sess.id}.md", data=md,
                mime="text/markdown", sha=hashlib.sha256(md).hexdigest())


@register("drive_delete")
def handle_delete(db: Session, payload: dict) -> None:
    row = db.execute(select(DriveFile).where(DriveFile.target_id == payload["target_id"])).scalar_one_or_none()
    if row is None or not row.drive_file_id:
        if row is not None:
            row.storage_status = "deleted"
        return
    if get_settings().drive_configured:
        drive.delete(row.drive_file_id)
    row.storage_status = "deleted"
    row.updated_at = utcnow()
    audit(db, "drive.deleted", actor_email="system", target_type=row.target_kind, target_id=row.target_id)


def retry_failed(db: Session) -> int:
    from ..jobs.queue import enqueue
    n = 0
    for doc in db.execute(select(Document).where(Document.storage_status == "failed", Document.deleted_at.is_(None))).scalars():
        enqueue(db, "drive_sync_document", {"document_id": str(doc.id)}, dedupe_key=f"drive:{doc.id}", max_attempts=6)
        doc.storage_status = "pending"
        n += 1
    return n
