"""Live turn analysis (fast model) — the interviewer's ears, NOT the evaluator."""

from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, ConfigDict, Field


class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")


class NewClaim(_M):
    text: str
    relates_to: str = Field("", description="id of the CV claim (C#) or earlier claim (IC#) this statement is about, "
                                            "if it is about the same thing; else ''")
    slot: str = Field("", description="comparable attribute, e.g. team_size, revenue_growth_pct, budget_inr")
    value: float | None = None
    unit: str = ""
    high_impact: bool = False
    supported_in_answer: bool = False


class CVReference(_M):
    claim_id: str
    consistency: Literal["consistent", "inconsistent", "unclear"] = "unclear"
    note: str = ""


class MemoryConflict(_M):
    memory_ref: str = Field(..., description="a memory id (M#) or claim id (C#/IC#) from the context")
    description: str
    severity: Literal["low", "medium", "high"] = "medium"


class TurnAnalysis(_M):
    intent: Literal["answer", "clarification_request", "repeat_request", "thinking_pause", "off_topic_question",
                    "refusal", "break_request", "end_request", "meta_question", "non_answer"] = "answer"
    addresses_question: Literal["fully", "partially", "no"] = "partially"
    answer_quality: Literal["strong", "adequate", "weak"] = "adequate"
    specificity: int = Field(1, ge=0, le=3)
    ownership_clarity: int = Field(1, ge=0, le=3)
    reasoning_present: bool = False
    quantified: bool = False
    structure_observed: str = ""
    knowledge_level: Literal["below", "at", "above", "n/a"] = "n/a"
    observed_signals: List[str] = Field(default_factory=list, description="expected-evidence items that WERE shown")
    gaps: List[str] = Field(default_factory=list, description="expected-evidence items still missing")
    probe_focus: Literal["specificity", "ownership", "reasoning", "outcome", "quantification", "reflection",
                         "tradeoff", "depth", "example", "none"] = "none"
    new_claims: List[NewClaim] = Field(default_factory=list)
    cv_references: List[CVReference] = Field(default_factory=list)
    conflicts: List[MemoryConflict] = Field(default_factory=list)
    summary: str = Field("", description="<= 25 words, neutral, what the candidate said")
