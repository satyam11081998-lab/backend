"""CandidateProfile — structured CV intelligence (spec §7)."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

ClaimType = Literal["leadership", "impact", "scale", "ownership", "technical", "award",
                    "tool", "education", "certification", "other"]
Seniority = Literal["intern", "entry", "mid", "senior", "lead", "executive", "unknown"]


class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Metric(_M):
    value: Optional[float] = None
    unit: str = Field("", description="%, INR, USD, people, users, x, days ...")
    direction: Literal["increase", "decrease", "absolute", "unknown"] = "unknown"
    raw: str = ""


class CVClaim(_M):
    id: str = Field(..., description="C1, C2 ... unique")
    text: str = Field(..., description="the claim, close to the CV wording")
    type: ClaimType = "other"
    experience_id: Optional[str] = None
    quantified: bool = False
    metrics: List[Metric] = Field(default_factory=list)
    slots: Dict[str, float] = Field(default_factory=dict,
                                    description="comparable numeric facts, e.g. team_size, budget_inr, users")
    ownership_language: Literal["personal", "team", "ambiguous"] = "ambiguous"
    vague: bool = False
    needs_verification: bool = True
    verification_priority: int = Field(2, ge=1, le=3, description="1 = probe first")


class Experience(_M):
    id: str = Field(..., description="X1, X2 ...")
    organization: str = ""
    title: str = ""
    start: str = Field("", description="as written, e.g. 'Jun 2021'")
    end: str = Field("", description="as written or 'Present'")
    is_current: bool = False
    location: str = ""
    function: str = ""
    responsibilities: List[str] = Field(default_factory=list)


class Education(_M):
    institution: str = ""
    degree: str = ""
    field: str = ""
    start: str = ""
    end: str = ""
    grade: str = ""


class Project(_M):
    name: str = ""
    description: str = ""
    role: str = ""
    technologies: List[str] = Field(default_factory=list)


class Certification(_M):
    name: str = ""
    issuer: str = ""
    year: str = ""


class Activity(_M):
    """Life beyond the job: positions of responsibility, clubs, competitions, sport, volunteering,
    awards outside work. Interviewers ask about these to see the person, not only the employee."""
    id: str = Field("", description="A1, A2 ...")
    kind: Literal["position_of_responsibility", "extracurricular", "competition", "sport", "volunteering",
                  "award", "creative", "other"] = "other"
    text: str = Field("", description="close to the CV wording")
    organization: str = ""


class Skills(_M):
    technical: List[str] = Field(default_factory=list)
    tools: List[str] = Field(default_factory=list)
    domain: List[str] = Field(default_factory=list)
    soft: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)


class ParseQuality(_M):
    confidence: Literal["high", "medium", "low"] = "medium"
    missing_sections: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class CandidateProfile(_M):
    headline: str = ""
    current_title: str = ""
    total_experience_years: Optional[float] = None
    experience_basis: Literal["computed_from_dates", "stated", "unknown"] = "unknown"
    seniority_estimate: Seniority = "unknown"
    functions: List[str] = Field(default_factory=list)
    industries: List[str] = Field(default_factory=list)
    experience: List[Experience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    projects: List[Project] = Field(default_factory=list)
    skills: Skills = Field(default_factory=Skills)
    certifications: List[Certification] = Field(default_factory=list)
    activities: List[Activity] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list, description="hobbies / interests as listed, short phrases")
    claims: List[CVClaim] = Field(default_factory=list)
    parse_quality: ParseQuality = Field(default_factory=ParseQuality)


class TimelineIssue(_M):
    type: Literal["gap", "overlap", "inconsistent_dates", "future_date", "unparseable_date"]
    detail: str
    refs: List[str] = Field(default_factory=list)
