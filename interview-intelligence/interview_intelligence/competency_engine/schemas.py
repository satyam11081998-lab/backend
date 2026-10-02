"""Schemas for role classification, competency mapping and rubrics."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Importance = Literal["critical", "high", "medium", "low"]


class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")


class FamilyCandidate(_M):
    family_id: str
    confidence: float = Field(0.5, ge=0, le=1)
    reason: str = ""


class RoleClassification(_M):
    primary_family: str = Field(..., description="an id from the provided taxonomy, or 'other'")
    sub_family: str = ""
    candidates: List[FamilyCandidate] = Field(default_factory=list)
    industry: str = Field("", description="an id from the provided industries list, or ''")
    is_hybrid: bool = False
    proposed_family_name: str = Field("", description="only when primary_family is 'other'")
    technical_role: bool = False
    rationale: str = ""


class MappedCompetency(_M):
    competency_id: str = Field(..., description="a canonical id from the library, or 'rs:<snake_case>' for a "
                                                "role-specific competency")
    name: str = ""
    parent: str = Field("", description="for rs: ids, the canonical parent id")
    importance: Importance = "medium"
    requirement_ids: List[str] = Field(default_factory=list)
    responsibility_ids: List[str] = Field(default_factory=list)
    sub_areas: List[str] = Field(default_factory=list,
                                 description="role-specific topics to probe, each justified by the JD")
    expected_depth: str = Field("", description="what 'good' means at this seniority, one sentence")
    cv_claim_ids: List[str] = Field(default_factory=list)
    cv_strength: Literal["strong", "partial", "none"] = "none"
    definition: str = Field("", description="required for rs: ids")


class CompetencyMapping(_M):
    competencies: List[MappedCompetency] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class RubricEntry(_M):
    competency_id: str
    what_good_looks_like: str = Field(..., description="role + seniority specific, 1-3 sentences")
    strong_signals: List[str] = Field(default_factory=list)
    weak_signals: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    expected_structure: str = ""
    must_not_infer: List[str] = Field(default_factory=list)
    band_notes: Optional[dict] = Field(None, description="optional {'exceptional','strong','moderate','needs_development'} "
                                                         "role-specific notes; canonical anchors still apply")


class RubricSet(_M):
    rubrics: List[RubricEntry] = Field(default_factory=list)
