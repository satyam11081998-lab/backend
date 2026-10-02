"""Evaluator + assessment QA outputs (spec §35–§37, §66–§68)."""

from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

EvidenceState = Literal["strong", "moderate", "weak", "contradictory", "not_sufficiently_tested"]


class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")


class CompetencyEvaluation(_M):
    competency_id: str
    evidence_state: EvidenceState
    score: Optional[int] = Field(None, ge=1, le=10, description="null when not_sufficiently_tested")
    rationale: str = Field("", description="2-4 sentences tying the score to cited evidence refs")
    evidence_refs: List[str] = Field(default_factory=list, description="E# refs from the supplied evidence ONLY")
    strengths: List[str] = Field(default_factory=list)
    gaps: List[str] = Field(default_factory=list)
    missing_evidence: List[str] = Field(default_factory=list, description="what was not shown / not tested")


class QAFinding(_M):
    competency_id: str = ""
    problem: Literal["score_unsupported", "polarity_mismatch", "protected_attribute", "style_bias",
                     "keyword_reward", "absence_as_weakness", "overconfident", "contradiction", "other"] = "other"
    detail: str = ""
    severity: Literal["low", "medium", "high"] = "medium"


class AssessmentQA(_M):
    findings: List[QAFinding] = Field(default_factory=list)
