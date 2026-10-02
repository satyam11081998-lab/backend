"""Feedback bundle (spec §39–§47)."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field

FEEDBACK_CATEGORIES = (
    "knowledge", "functional_depth", "technical_depth", "role_understanding", "structured_thinking",
    "problem_solving", "analytical_reasoning", "commercial_reasoning", "communication", "articulation",
    "conciseness", "specificity", "ownership", "quantification", "evidence", "behavioral_examples",
    "leadership", "stakeholder_management", "situational_judgment", "domain_terminology", "cv_defense",
    "motivation", "company_understanding", "answer_consistency", "case_skills",
)
Category = Literal[FEEDBACK_CATEGORIES]  # type: ignore[valid-type]


class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")


class Strength(_M):
    title: str
    competency_ids: List[str] = Field(default_factory=list)
    evidence_refs: List[str] = Field(default_factory=list)
    why_it_matters: str = ""


class DevelopmentArea(_M):
    category: Category  # type: ignore[valid-type]
    severity: Literal["high", "medium", "low"] = "medium"
    title: str
    competency_ids: List[str] = Field(default_factory=list)
    observed_problem: str = Field(..., description="what exactly happened, specific")
    example_refs: List[str] = Field(default_factory=list, description="exchange refs X# where it happened")
    example_quote: str = ""
    why_it_matters: str = ""
    what_to_do: str = ""
    practice: str = Field("", description="a concrete drill with a count")
    measured_basis: List[str] = Field(default_factory=list, description="metric ids used for any number stated")


class LearnedItem(_M):
    text: str
    refs: List[str] = Field(default_factory=list)


class InterviewerLearned(_M):
    strong_signals: List[LearnedItem] = Field(default_factory=list)
    weak_signals: List[LearnedItem] = Field(default_factory=list)
    unproven_claims: List[LearnedItem] = Field(default_factory=list)
    missing_evidence: List[LearnedItem] = Field(default_factory=list)
    potential_concerns: List[LearnedItem] = Field(default_factory=list)


class NextQuestion(_M):
    question: str
    gap: str = Field(..., description="the evidence gap that makes this the natural next question")
    refs: List[str] = Field(default_factory=list)


class PrepItem(_M):
    action: str
    count: int = Field(1, ge=1, le=10)
    competency_ids: List[str] = Field(default_factory=list)
    linked_development: List[int] = Field(default_factory=list, description="indexes into development_areas")
    how: str = ""


class PrepPlan(_M):
    headline: str = ""
    items: List[PrepItem] = Field(default_factory=list)
    topics_to_revise: List[str] = Field(default_factory=list)
    reattempt_competencies: List[str] = Field(default_factory=list)


class FeedbackBundle(_M):
    executive_assessment: str
    strengths: List[Strength] = Field(default_factory=list)
    development_areas: List[DevelopmentArea] = Field(default_factory=list)
    interviewer_learned: InterviewerLearned = Field(default_factory=InterviewerLearned)
    next_questions: List[NextQuestion] = Field(default_factory=list)
    preparation_plan: PrepPlan = Field(default_factory=PrepPlan)


class FeedbackQAItem(_M):
    index: int
    passed: bool = True
    problems: List[str] = Field(default_factory=list)


class FeedbackQA(_M):
    development_areas: List[FeedbackQAItem] = Field(default_factory=list)
    executive_assessment_ok: bool = True
    executive_problems: List[str] = Field(default_factory=list)
