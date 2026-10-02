"""Company intelligence with provenance (spec §60, §61, §84, §85).

Sources implemented: JD-derived facts and user-provided notes. Official-site / public-report
research is a pluggable step behind `company_intel.web_research` (off; not built in this
release — see docs/BUILD_STATUS.md). Nothing inferred is ever presented as fact: every fact
carries its source type, and the JD always outranks company context."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..ai.guard import scan_injection
from ..db.models import CompanyProfile, utcnow
from .schemas import SOURCE_PRIORITY, CompanyFact


def _now() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def facts_from_jd(role: dict) -> List[CompanyFact]:
    out: List[CompanyFact] = []
    company = ((role.get("identity") or {}).get("company") or "").strip()
    for i, f in enumerate(role.get("company_facts_in_jd") or [], start=1):
        out.append(CompanyFact(id=f"J{i}", fact=str(f)[:400], source_type="jd_derived",
                               source=f"Job description{(' — ' + company) if company else ''}", retrieved_at=_now(),
                               confidence="high", relevance="high", used_for=["question_context"]))
    return out


def facts_from_notes(notes: str) -> List[CompanyFact]:
    out: List[CompanyFact] = []
    if not notes.strip():
        return out
    if scan_injection(notes):
        # Keep the candidate's notes usable but never as authority: low confidence, flagged.
        conf = "low"
    else:
        conf = "medium"
    parts = [p.strip(" -•\t") for p in re.split(r"[\n;]+", notes) if len(p.strip()) > 8][:10]
    for i, p in enumerate(parts, start=1):
        out.append(CompanyFact(id=f"U{i}", fact=p[:300], source_type="user_provided", source="Your notes",
                               retrieved_at=_now(), confidence=conf, relevance="medium", used_for=["question_context"]))
    return out


def build_profile(db: Session, user_id: uuid.UUID, company_name: str, role: dict, notes: str) -> Optional[CompanyProfile]:
    name = (company_name or (role.get("identity") or {}).get("company") or "").strip()
    facts = facts_from_jd(role) + facts_from_notes(notes)
    if not name and not facts:
        return None
    facts.sort(key=lambda f: SOURCE_PRIORITY[f.source_type])
    prof = db.execute(select(CompanyProfile).where(CompanyProfile.user_id == user_id,
                                                   CompanyProfile.company_name_lc == name.lower())).scalars().first()
    if prof is None:
        prof = CompanyProfile(user_id=user_id, company_name=name or "Unspecified", company_name_lc=name.lower())
        db.add(prof)
    prof.facts = [f.model_dump() for f in facts]
    prof.updated_at = utcnow()
    db.flush()
    return prof
