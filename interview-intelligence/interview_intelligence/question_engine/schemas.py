"""Question metadata (spec §18–§19) and generator / QA outputs."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

QuestionType = Literal["behavioral", "situational", "functional", "technical", "case", "cv_deep_dive",
                       "motivation", "company", "estimation", "closing", "intro", "pressure", "personal", "awareness"]


class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")


class SelectionReason(_M):
    requirement_ids: List[str] = Field(default_factory=list)
    claim_ids: List[str] = Field(default_factory=list)
    company_fact_ids: List[str] = Field(default_factory=list)
    news_ids: List[str] = Field(default_factory=list)
    activity_refs: List[str] = Field(default_factory=list)  # hobbies / activities from the CV it asks about
    explanation: str = ""


class PlannedQuestion(_M):
    qid: str = ""
    origin: Literal["curated", "archetype", "generated", "cv_specific", "fixed"] = "generated"
    archetype_id: str = ""
    curated_id: str = ""
    section_kind: str = ""
    competency_ids: List[str] = Field(default_factory=list)
    sub_competency: str = ""
    question_type: QuestionType = "behavioral"
    difficulty: int = Field(3, ge=1, le=5)
    seniority: str = ""
    technical_depth: int = Field(1, ge=0, le=5)
    ambiguity: int = Field(1, ge=0, le=5)
    text: str
    intent: str = Field("", description="what we are testing")
    expected_evidence: List[str] = Field(default_factory=list)
    strong_signals: List[str] = Field(default_factory=list)
    weak_signals: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    must_not_infer: List[str] = Field(default_factory=list)
    probe_tree: List[str] = Field(default_factory=list, description="ordered follow-up foci/questions")
    follow_ups: List[str] = Field(default_factory=list)
    prerequisites: List[str] = Field(default_factory=list)
    expected_duration_s: int = 180
    selection_reason: SelectionReason = Field(default_factory=SelectionReason)
    optional: bool = False


class GeneratedQuestion(_M):
    text: str
    archetype_id: str = ""
    competency_ids: List[str] = Field(default_factory=list)
    sub_competency: str = ""
    question_type: QuestionType = "behavioral"
    difficulty: int = Field(3, ge=1, le=5)
    intent: str = ""
    expected_evidence: List[str] = Field(default_factory=list)
    strong_signals: List[str] = Field(default_factory=list)
    weak_signals: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    must_not_infer: List[str] = Field(default_factory=list)
    probe_tree: List[str] = Field(default_factory=list)
    requirement_ids: List[str] = Field(default_factory=list)
    claim_ids: List[str] = Field(default_factory=list)
    news_ids: List[str] = Field(default_factory=list, description="N# of the news item the question uses, if any")
    activity_refs: List[str] = Field(default_factory=list, description="A# activity or the interest it asks about")
    why_this_question: str = ""


class GeneratedQuestions(_M):
    questions: List[GeneratedQuestion] = Field(default_factory=list)


class QAIssue(_M):
    qid: str = ""
    section: str = ""
    issue: Literal["irrelevant", "generic", "duplicate", "wrong_level", "wrong_domain", "unanswerable",
                   "leading", "answer_leak", "unfair", "unbalanced", "other"] = "other"
    detail: str = ""
    severity: Literal["low", "medium", "high"] = "medium"


class BlueprintQA(_M):
    passed: bool = True
    issues: List[QAIssue] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class ArchetypeSpec(_M):
    id: str
    name: str
    question_type: QuestionType
    competency_tags: List[str] = Field(default_factory=list)
    pattern: str
    intent: str
    expected_evidence: List[str] = Field(default_factory=list)
    strong_signals: List[str] = Field(default_factory=list)
    weak_signals: List[str] = Field(default_factory=list)
    red_flags: List[str] = Field(default_factory=list)
    probe_tree: List[str] = Field(default_factory=list)
    must_not_infer: List[str] = Field(default_factory=list)
    difficulty_range: List[int] = Field(default_factory=lambda: [2, 4])
    modes: List[str] = Field(default_factory=list)
    expected_duration_s: int = 180
    extra: Optional[Dict[str, str]] = None
