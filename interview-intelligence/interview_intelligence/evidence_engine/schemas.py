"""Evidence extraction output (spec §24–§25)."""

from __future__ import annotations

from typing import Dict, List, Literal

from pydantic import BaseModel, ConfigDict, Field

EvidenceType = Literal["claim", "action", "reasoning", "outcome", "quantification", "reflection", "knowledge",
                       "structure", "communication", "judgment", "tradeoff", "ownership"]
DIMENSIONS = ("relevance", "structure", "clarity", "conciseness", "specificity", "ownership", "reasoning",
              "evidence", "quantification", "terminology", "business_relevance", "depth", "reflection",
              "consistency")


class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ExtractedEvidence(_M):
    competency_id: str
    sub_competency: str = ""
    type: EvidenceType = "claim"
    polarity: Literal["positive", "negative", "neutral"] = "neutral"
    strength: Literal["strong", "moderate", "weak"] = "moderate"
    quote: str = Field(..., description="VERBATIM words from the CANDIDATE's messages, 5-40 words")
    interpretation: str = Field("", description="what this shows, one sentence, no speculation")
    ownership: Literal["personal", "team", "unclear", "n/a"] = "unclear"
    cv_consistency: Literal["consistent", "inconsistent", "not_in_cv", "n/a"] = "n/a"
    confidence: float = Field(0.6, ge=0, le=1)


class DimensionRating(_M):
    applicable: bool = True
    rating: int | None = Field(None, ge=1, le=5)
    note: str = ""


class ExchangeEvidence(_M):
    claim: str = ""
    evidence: str = ""
    action: str = ""
    reasoning: str = ""
    outcome: str = ""
    quantification: str = ""
    reflection: str = ""
    missing: List[str] = Field(default_factory=list)
    contradictions: List[str] = Field(default_factory=list)
    items: List[ExtractedEvidence] = Field(default_factory=list)
    dimensions: Dict[str, DimensionRating] = Field(default_factory=dict)
    what_worked: List[str] = Field(default_factory=list)
    what_was_missing: List[str] = Field(default_factory=list)
    interviewer_was_looking_for: str = ""
    better_answer_direction: str = ""
    exposing_follow_up: str = Field("", description="the follow-up question that exposed a gap, if any (quote it)")
