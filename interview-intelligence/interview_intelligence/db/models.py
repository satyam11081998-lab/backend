"""Interview Intelligence data model — schema `interview_intel` ONLY.

Design notes
* Every table lives in the dedicated schema. `tests/test_isolation_static.py` asserts it.
* Structured columns for anything we filter, join, count or aggregate; JSON only for
  genuinely flexible payloads (parsed profiles, blueprints, reports, model outputs).
* UUID primary keys generated in Python so the same code runs on Postgres and on the
  SQLite engine the unit tests use.
* Status columns are guarded by CHECK constraints that mirror the Python enums.
* This module is the source of truth for the SQL migration
  (`scripts/generate_migration.py` renders `migrations/0001_interview_intel.sql`).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    MetaData,
    Numeric,
    String,
    Text,
    TypeDecorator,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

SCHEMA = "interview_intel"

JSONType = JSON().with_variant(JSONB(), "postgresql")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def new_id() -> uuid.UUID:
    return uuid.uuid4()


def _check_in(col: str, values) -> str:
    quoted = ", ".join(f"'{v}'" for v in values)
    return f"{col} IN ({quoted})"


class Base(DeclarativeBase):
    metadata = MetaData(
        schema=SCHEMA,
        naming_convention={
            "ix": "ix_%(table_name)s_%(column_0_N_name)s",
            "uq": "uq_%(table_name)s_%(column_0_N_name)s",
            "ck": "ck_%(table_name)s_%(constraint_name)s",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        },
    )


class _UTCDateTime(TypeDecorator):
    """timestamptz on Postgres. SQLite (dev/tests) drops the offset, so values are normalised to
    aware UTC on the way in and out — Python-side comparisons with utcnow() then behave the
    same on both databases."""
    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value

    def process_result_value(self, value, dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value


TS = _UTCDateTime()

# --------------------------------------------------------------------------------------
# Enumerations (mirrored by CHECK constraints)
# --------------------------------------------------------------------------------------
SESSION_STATUSES = (
    "created", "uploading", "analyzing", "ready", "active", "paused",
    "completed", "abandoned", "expired", "failed",
)
ASSESSMENT_STATUSES = ("none", "pending", "processing", "ready", "partial", "failed")
DOC_KINDS = ("cv", "jd")
PARSE_STATUSES = ("pending", "parsed", "unreadable", "failed")
ANALYSIS_STATUSES = ("pending", "running", "ready", "failed", "unreadable")
STORAGE_STATUSES = ("pending", "uploading", "synced", "failed", "disabled", "deleted")
GRANT_STATUSES = ("enabled", "disabled")
JOB_STATUSES = ("queued", "running", "succeeded", "failed", "dead")
MESSAGE_ROLES = ("interviewer", "candidate", "system")
EVIDENCE_STATES = ("strong", "moderate", "weak", "contradictory", "not_sufficiently_tested")


# --------------------------------------------------------------------------------------
# Identity, access, configuration
# --------------------------------------------------------------------------------------
class User(Base):
    """II-local registry of people who have used II. Keyed by the MECE user id from the
    signed assertion. II never reads MECE's own users table."""

    __tablename__ = "users"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    email: Mapped[str] = mapped_column(String(320), default="")
    email_lc: Mapped[str] = mapped_column(String(320), default="", index=True)
    last_tier: Mapped[str] = mapped_column(String(16), default="free")
    last_entitled: Mapped[bool] = mapped_column(Boolean, default=False)
    last_access_via: Mapped[str | None] = mapped_column(String(32), nullable=True)
    drive_folder_key: Mapped[str | None] = mapped_column(String(64), nullable=True, unique=True)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    last_seen_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    deleted_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)


class AccessGrant(Base):
    """Admin-managed test/comp access by email (spec §4). Not hard-coded anywhere."""

    __tablename__ = "access_grants"
    __table_args__ = (CheckConstraint(_check_in("status", GRANT_STATUSES), name="status"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    email_lc: Mapped[str] = mapped_column(String(320), unique=True)
    status: Mapped[str] = mapped_column(String(16), default="enabled")
    grant_type: Mapped[str] = mapped_column(String(16), default="test")
    note: Mapped[str] = mapped_column(Text, default="")
    granted_by: Mapped[str] = mapped_column(String(320), default="")
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    expires_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)


class SystemConfig(Base):
    __tablename__ = "system_config"
    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[dict] = mapped_column(JSONType)
    updated_by: Mapped[str] = mapped_column(String(320), default="")
    updated_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow, index=True)
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    actor_email: Mapped[str] = mapped_column(String(320), default="")
    action: Mapped[str] = mapped_column(String(64), index=True)
    target_type: Mapped[str] = mapped_column(String(64), default="")
    target_id: Mapped[str] = mapped_column(String(128), default="")
    meta: Mapped[dict] = mapped_column(JSONType, default=dict)


# --------------------------------------------------------------------------------------
# Documents and storage
# --------------------------------------------------------------------------------------
class Document(Base):
    __tablename__ = "documents"
    __table_args__ = (
        CheckConstraint(_check_in("kind", DOC_KINDS), name="kind"),
        CheckConstraint(_check_in("parse_status", PARSE_STATUSES), name="parse_status"),
        CheckConstraint(_check_in("storage_status", STORAGE_STATUSES), name="storage_status"),
        Index("ix_documents_user_kind", "user_id", "kind"),
        Index("ix_documents_dedupe", "user_id", "kind", "sha256"),
        Index("ix_documents_text_hash", "user_id", "kind", "text_sha256"),
    )
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    kind: Mapped[str] = mapped_column(String(8))
    source: Mapped[str] = mapped_column(String(16), default="upload")  # upload | paste
    label: Mapped[str] = mapped_column(String(200), default="")
    file_name: Mapped[str] = mapped_column(String(255), default="")
    mime_type: Mapped[str] = mapped_column(String(128), default="")
    ext: Mapped[str] = mapped_column(String(8), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    sha256: Mapped[str] = mapped_column(String(64))
    # Hash of the normalised extracted text: a re-saved file with identical content gets a new
    # document row (its bytes differ) but reuses the existing analysis — same user only.
    text_sha256: Mapped[str] = mapped_column(String(64), default="")
    version: Mapped[int] = mapped_column(Integer, default=1)  # per (user, kind) — "CV v3"
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    text_chars: Mapped[int] = mapped_column(Integer, default=0)
    text_quality: Mapped[dict] = mapped_column(JSONType, default=dict)
    parse_status: Mapped[str] = mapped_column(String(16), default="pending")
    parse_error: Mapped[str] = mapped_column(Text, default="")
    injection_flags: Mapped[list] = mapped_column(JSONType, default=list)
    storage_status: Mapped[str] = mapped_column(String(16), default="pending")
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    deleted_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)


class DocumentContent(Base):
    """Encrypted payloads, split from `documents` so list queries never load bytes."""

    __tablename__ = "document_contents"
    document_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    blob_enc: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    text_enc: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    redacted_text_enc: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    key_id: Mapped[str] = mapped_column(String(32), default="k1")
    purged_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)


class DocumentAnalysis(Base):
    """Cached structured analysis keyed by (document, analysis_version) — spec §96."""

    __tablename__ = "document_analyses"
    __table_args__ = (
        UniqueConstraint("document_id", "analysis_version"),
        CheckConstraint(_check_in("status", ANALYSIS_STATUSES), name="status"),
    )
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"))
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(8))
    sha256: Mapped[str] = mapped_column(String(64))
    analysis_version: Mapped[str] = mapped_column(String(32))
    prompt_version: Mapped[str] = mapped_column(String(32), default="")
    model: Mapped[str] = mapped_column(String(96), default="")
    status: Mapped[str] = mapped_column(String(16), default="pending")
    result: Mapped[dict] = mapped_column(JSONType, default=dict)
    quality: Mapped[dict] = mapped_column(JSONType, default=dict)
    error: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class DriveFolder(Base):
    __tablename__ = "drive_folders"
    __table_args__ = (UniqueConstraint("scope_key", "kind"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    scope_key: Mapped[str] = mapped_column(String(64))  # "global" or a user id
    kind: Mapped[str] = mapped_column(String(16))  # users_root | user | cv | jd | interviews
    folder_id: Mapped[str] = mapped_column(String(128))
    parent_folder_id: Mapped[str] = mapped_column(String(128), default="")
    name: Mapped[str] = mapped_column(String(128), default="")
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class DriveFile(Base):
    __tablename__ = "drive_files"
    __table_args__ = (
        CheckConstraint(_check_in("storage_status", STORAGE_STATUSES), name="storage_status"),
        UniqueConstraint("target_kind", "target_id"),
    )
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    target_kind: Mapped[str] = mapped_column(String(16))  # cv | jd | report
    target_id: Mapped[str] = mapped_column(String(64))  # document id or session id
    drive_file_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    drive_folder_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    file_name: Mapped[str] = mapped_column(String(255), default="")
    mime_type: Mapped[str] = mapped_column(String(128), default="")
    sha256: Mapped[str] = mapped_column(String(64), default="")
    storage_status: Mapped[str] = mapped_column(String(16), default="pending")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str] = mapped_column(Text, default="")
    uploaded_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


# --------------------------------------------------------------------------------------
# Role / company intelligence
# --------------------------------------------------------------------------------------
class CompanyProfile(Base):
    __tablename__ = "company_profiles"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    company_name: Mapped[str] = mapped_column(String(200))
    company_name_lc: Mapped[str] = mapped_column(String(200), index=True)
    facts: Mapped[list] = mapped_column(JSONType, default=list)  # provenance-tagged facts
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class GeneratedQuestion(Base):
    """Question library growth: generated items are kept for reuse, dedupe and analytics."""

    __tablename__ = "questions"
    __table_args__ = (UniqueConstraint("text_hash"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    origin: Mapped[str] = mapped_column(String(16))  # generated | cv_specific
    role_family: Mapped[str] = mapped_column(String(64), index=True)
    competency_ids: Mapped[list] = mapped_column(JSONType, default=list)
    question_type: Mapped[str] = mapped_column(String(32))
    difficulty: Mapped[int] = mapped_column(Integer, default=3)
    text: Mapped[str] = mapped_column(Text)
    text_hash: Mapped[str] = mapped_column(String(64))
    meta: Mapped[dict] = mapped_column(JSONType, default=dict)
    quality_status: Mapped[str] = mapped_column(String(16), default="unreviewed")
    times_selected: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


# --------------------------------------------------------------------------------------
# Interview sessions
# --------------------------------------------------------------------------------------
class InterviewSession(Base):
    __tablename__ = "interview_sessions"
    __table_args__ = (
        CheckConstraint(_check_in("status", SESSION_STATUSES), name="status"),
        CheckConstraint(_check_in("assessment_status", ASSESSMENT_STATUSES), name="assessment_status"),
        Index("ix_interview_sessions_user_status", "user_id", "status"),
        Index("ix_interview_sessions_user_created", "user_id", "created_at"),
    )
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    status: Mapped[str] = mapped_column(String(16), default="created")
    status_reason: Mapped[str] = mapped_column(Text, default="")
    cv_document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id"))
    jd_document_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("documents.id"))
    company_profile_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("company_profiles.id"), nullable=True
    )
    source_session_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    config: Mapped[dict] = mapped_column(JSONType, default=dict)
    mode: Mapped[str] = mapped_column(String(32), default="mixed", index=True)
    difficulty: Mapped[str] = mapped_column(String(16), default="medium")
    duration_minutes: Mapped[int] = mapped_column(Integer, default=45)
    role_family: Mapped[str] = mapped_column(String(64), default="")
    role_title: Mapped[str] = mapped_column(String(200), default="")
    company_name: Mapped[str] = mapped_column(String(200), default="")
    pre_interview_summary: Mapped[dict] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    started_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)
    last_activity_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    paused_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)
    ended_reason: Mapped[str] = mapped_column(String(32), default="")
    active_seconds: Mapped[int] = mapped_column(Integer, default=0)
    assessment_status: Mapped[str] = mapped_column(String(16), default="none")
    assessment_confidence: Mapped[str] = mapped_column(String(16), default="")
    cost_usd: Mapped[float] = mapped_column(Numeric(12, 6), default=0)
    versions: Mapped[dict] = mapped_column(JSONType, default=dict)
    deleted_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)


class InterviewBlueprint(Base):
    __tablename__ = "interview_blueprints"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"), unique=True
    )
    role_profile: Mapped[dict] = mapped_column(JSONType, default=dict)
    competency_model: Mapped[dict] = mapped_column(JSONType, default=dict)
    rubric: Mapped[dict] = mapped_column(JSONType, default=dict)
    rubric_hash: Mapped[str] = mapped_column(String(64), default="")
    blueprint: Mapped[dict] = mapped_column(JSONType, default=dict)
    qa_report: Mapped[dict] = mapped_column(JSONType, default=dict)
    engine_versions: Mapped[dict] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class InterviewState(Base):
    __tablename__ = "interview_states"
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"), primary_key=True
    )
    version: Mapped[int] = mapped_column(Integer, default=0)
    state: Mapped[dict] = mapped_column(JSONType, default=dict)
    updated_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class InterviewExchange(Base):
    """One planned question plus its follow-ups — the unit of question-level analysis."""

    __tablename__ = "interview_exchanges"
    __table_args__ = (UniqueConstraint("session_id", "seq"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer)
    planned_qid: Mapped[str] = mapped_column(String(32), default="")
    section_id: Mapped[str] = mapped_column(String(32), default="")
    question_text: Mapped[str] = mapped_column(Text, default="")
    question_meta: Mapped[dict] = mapped_column(JSONType, default=dict)
    competency_ids: Mapped[list] = mapped_column(JSONType, default=list)
    difficulty: Mapped[int] = mapped_column(Integer, default=3)
    status: Mapped[str] = mapped_column(String(16), default="open")  # open|closed|skipped
    probes: Mapped[int] = mapped_column(Integer, default=0)
    live_signals: Mapped[list] = mapped_column(JSONType, default=list)
    metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    evidence_status: Mapped[str] = mapped_column(String(16), default="pending")
    analysis: Mapped[dict] = mapped_column(JSONType, default=dict)
    started_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    closed_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)


class InterviewMessage(Base):
    __tablename__ = "interview_messages"
    __table_args__ = (
        UniqueConstraint("session_id", "seq"),
        UniqueConstraint("session_id", "client_turn_id"),
        CheckConstraint(_check_in("role", MESSAGE_ROLES), name="role"),
    )
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"), index=True
    )
    seq: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(16))
    kind: Mapped[str] = mapped_column(String(16), default="text")
    content: Mapped[str] = mapped_column(Text)
    exchange_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    action: Mapped[str] = mapped_column(String(32), default="")
    client_turn_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    meta: Mapped[dict] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class InterviewEvent(Base):
    __tablename__ = "interview_events"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"), index=True
    )
    type: Mapped[str] = mapped_column(String(48))
    payload: Mapped[dict] = mapped_column(JSONType, default=dict)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class Claim(Base):
    """CV claims selected for investigation + claims made during the interview."""

    __tablename__ = "claims"
    __table_args__ = (UniqueConstraint("session_id", "claim_key"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"), index=True
    )
    claim_key: Mapped[str] = mapped_column(String(16))  # C3 (cv) / IC2 (interview)
    source: Mapped[str] = mapped_column(String(16))  # cv | interview
    text: Mapped[str] = mapped_column(Text)
    claim_type: Mapped[str] = mapped_column(String(32), default="other")
    slots: Mapped[dict] = mapped_column(JSONType, default=dict)
    priority: Mapped[int] = mapped_column(Integer, default=2)
    verification_status: Mapped[str] = mapped_column(String(32), default="unverified")
    exchange_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


# --------------------------------------------------------------------------------------
# Assessment
# --------------------------------------------------------------------------------------
class EvidenceItem(Base):
    __tablename__ = "evidence_items"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"), index=True
    )
    exchange_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("interview_exchanges.id", ondelete="CASCADE"), index=True
    )
    ref: Mapped[str] = mapped_column(String(16))  # E12 — what the evaluator cites
    competency_id: Mapped[str] = mapped_column(String(96), index=True)
    sub_competency: Mapped[str] = mapped_column(String(160), default="")
    evidence_type: Mapped[str] = mapped_column(String(32))
    polarity: Mapped[str] = mapped_column(String(16))
    strength: Mapped[str] = mapped_column(String(16))
    quote: Mapped[str] = mapped_column(Text, default="")
    quote_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    interpretation: Mapped[str] = mapped_column(Text, default="")
    ownership: Mapped[str] = mapped_column(String(16), default="unclear")
    cv_consistency: Mapped[str] = mapped_column(String(16), default="n/a")
    confidence: Mapped[float] = mapped_column(Float, default=0.5)
    extractor_version: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class CompetencyAssessment(Base):
    __tablename__ = "competency_assessments"
    __table_args__ = (
        UniqueConstraint("session_id", "competency_id"),
        CheckConstraint(_check_in("evidence_state", EVIDENCE_STATES), name="evidence_state"),
    )
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"), index=True
    )
    competency_id: Mapped[str] = mapped_column(String(96))
    canonical_competency_id: Mapped[str] = mapped_column(String(96))
    name: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(32))
    importance: Mapped[str] = mapped_column(String(16))
    evidence_state: Mapped[str] = mapped_column(String(32))
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    band: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[str] = mapped_column(String(16))
    confidence_basis: Mapped[dict] = mapped_column(JSONType, default=dict)
    rationale: Mapped[str] = mapped_column(Text, default="")
    evidence_refs: Mapped[list] = mapped_column(JSONType, default=list)
    strengths: Mapped[list] = mapped_column(JSONType, default=list)
    gaps: Mapped[list] = mapped_column(JSONType, default=list)
    missing_evidence: Mapped[list] = mapped_column(JSONType, default=list)
    anomaly_flags: Mapped[list] = mapped_column(JSONType, default=list)
    evaluator_version: Mapped[str] = mapped_column(String(32), default="")
    rubric_hash: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class FeedbackItem(Base):
    __tablename__ = "feedback_items"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(16))  # strength | development
    category: Mapped[str] = mapped_column(String(48), index=True)
    severity: Mapped[str] = mapped_column(String(16), default="medium")
    competency_ids: Mapped[list] = mapped_column(JSONType, default=list)
    title: Mapped[str] = mapped_column(String(300))
    body: Mapped[dict] = mapped_column(JSONType, default=dict)
    qa_status: Mapped[str] = mapped_column(String(16), default="passed")
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class Report(Base):
    __tablename__ = "reports"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    session_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("interview_sessions.id", ondelete="CASCADE"), unique=True
    )
    status: Mapped[str] = mapped_column(String(16))  # ready | partial
    report: Mapped[dict] = mapped_column(JSONType, default=dict)
    report_version: Mapped[str] = mapped_column(String(32), default="")
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


class CompetencyHistory(Base):
    __tablename__ = "competency_history"
    __table_args__ = (
        Index("ix_competency_history_user_comp", "user_id", "canonical_competency_id"),
        UniqueConstraint("session_id", "competency_id"),
    )
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    session_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("interview_sessions.id", ondelete="CASCADE"))
    competency_id: Mapped[str] = mapped_column(String(96))
    canonical_competency_id: Mapped[str] = mapped_column(String(96))
    role_family: Mapped[str] = mapped_column(String(64), default="")
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    evidence_state: Mapped[str] = mapped_column(String(32))
    confidence: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)


# --------------------------------------------------------------------------------------
# Observability, jobs, evaluation
# --------------------------------------------------------------------------------------
class ModelRun(Base):
    __tablename__ = "model_runs"
    __table_args__ = (Index("ix_model_runs_stage_created", "stage", "created_at"),)
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow, index=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    session_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True, index=True)
    stage: Mapped[str] = mapped_column(String(48))
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(96))
    prompt_id: Mapped[str] = mapped_column(String(64), default="")
    prompt_version: Mapped[str] = mapped_column(String(32), default="")
    schema_version: Mapped[str] = mapped_column(String(32), default="")
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cost_usd: Mapped[float] = mapped_column(Numeric(12, 6), default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(24))  # ok|repaired|invalid_output|error|timeout|budget
    attempt: Mapped[int] = mapped_column(Integer, default=1)
    error: Mapped[str] = mapped_column(Text, default="")
    validation: Mapped[dict] = mapped_column(JSONType, default=dict)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        CheckConstraint(_check_in("status", JOB_STATUSES), name="status"),
        Index("ix_jobs_status_run_after", "status", "run_after"),
        UniqueConstraint("dedupe_key"),
    )
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    kind: Mapped[str] = mapped_column(String(48))
    payload: Mapped[dict] = mapped_column(JSONType, default=dict)
    status: Mapped[str] = mapped_column(String(16), default="queued")
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    run_after: Mapped[datetime] = mapped_column(TS, default=utcnow)
    locked_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)
    locked_by: Mapped[str] = mapped_column(String(64), default="")
    last_error: Mapped[str] = mapped_column(Text, default="")
    dedupe_key: Mapped[str | None] = mapped_column(String(160), nullable=True)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(TS, nullable=True)


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=new_id)
    created_at: Mapped[datetime] = mapped_column(TS, default=utcnow)
    kind: Mapped[str] = mapped_column(String(24))  # golden | calibration | regression
    evaluator_version: Mapped[str] = mapped_column(String(32))
    prompt_versions: Mapped[dict] = mapped_column(JSONType, default=dict)
    dataset_version: Mapped[str] = mapped_column(String(32), default="")
    metrics: Mapped[dict] = mapped_column(JSONType, default=dict)
    results: Mapped[list] = mapped_column(JSONType, default=list)
    baseline_run_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, nullable=True)
    notes: Mapped[str] = mapped_column(Text, default="")


ALL_TABLES = [t for t in Base.metadata.sorted_tables]
