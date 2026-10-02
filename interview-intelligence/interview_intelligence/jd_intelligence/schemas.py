"""RoleProfile — structured JD intelligence (spec §9). Built to say 'unknown' rather than guess."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ReqCategory = Literal["education", "experience", "technical_skill", "functional_skill", "business_knowledge",
                      "tool", "certification", "communication", "leadership", "analytical", "commercial",
                      "domain", "behavioral", "other"]
Seniority = Literal["intern", "entry", "mid", "senior", "lead", "executive", "unknown"]


class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")


class JobIdentity(_M):
    company: str = ""
    title: str = ""
    role_name: str = ""
    function: str = ""
    business_unit: str = ""
    geography: str = ""
    employment_type: str = ""


class SeniorityAssessment(_M):
    level: Seniority = "unknown"
    confidence: Literal["high", "medium", "low"] = "low"
    basis: str = Field("", description="quote or reason; empty if unknown")
    years_required_min: Optional[float] = None
    years_required_max: Optional[float] = None


class Requirement(_M):
    id: str = Field(..., description="R1, R2 ... unique")
    text: str
    category: ReqCategory = "other"
    importance: Literal["must", "should", "nice"] = "should"
    explicit: bool = Field(True, description="false if implied rather than stated")


class Responsibility(_M):
    id: str = Field(..., description="RS1, RS2 ...")
    text: str
    kind: Literal["daily", "major", "periodic"] = "major"
    decision_making: bool = False
    stakeholders: List[str] = Field(default_factory=list)
    kpis: List[str] = Field(default_factory=list)


class Keyword(_M):
    term: str
    category: str = ""
    weight: Literal["high", "medium", "low"] = "medium"


class ImpliedCompetency(_M):
    name: str
    basis: List[str] = Field(default_factory=list, description="R/RS ids that imply it")


class JDQuality(_M):
    specificity: Literal["high", "medium", "low"] = "medium"
    contradictions: List[str] = Field(default_factory=list)
    ambiguities: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class RoleProfile(_M):
    identity: JobIdentity = Field(default_factory=JobIdentity)
    seniority: SeniorityAssessment = Field(default_factory=SeniorityAssessment)
    summary: str = ""
    requirements: List[Requirement] = Field(default_factory=list)
    responsibilities: List[Responsibility] = Field(default_factory=list)
    stakeholders: List[str] = Field(default_factory=list)
    kpis: List[str] = Field(default_factory=list)
    keywords: List[Keyword] = Field(default_factory=list)
    implied_competencies: List[ImpliedCompetency] = Field(default_factory=list)
    domain_requirements: List[str] = Field(default_factory=list)
    seniority_expectations: List[str] = Field(default_factory=list)
    company_facts_in_jd: List[str] = Field(default_factory=list,
                                           description="facts about the company stated in the JD itself")
    quality: JDQuality = Field(default_factory=JDQuality)
