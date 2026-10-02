"""Deterministic evaluation gates (docs/H_SCORING_MODEL.md §3) and confidence (§4).

The model proposes; these rules decide what is allowed to reach a candidate. They make it
impossible to (a) score without evidence, (b) score an untested area as weak, (c) give a
high score without strong evidence, (d) cite evidence that does not exist."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .schemas import CompetencyEvaluation

BANDS = [(9, "Exceptional"), (7, "Strong"), (5, "Moderate"), (1, "Needs development")]
STATE_FOR_SCORE = [(7, "strong"), (5, "moderate"), (1, "weak")]


def band_for(score: Optional[int]) -> str:
    if score is None:
        return "Insufficient evidence"
    for lo, name in BANDS:
        if score >= lo:
            return name
    return "Needs development"


def state_for(score: int) -> str:
    for lo, st in STATE_FOR_SCORE:
        if score >= lo:
            return st
    return "weak"


@dataclass
class EvidenceView:
    ref: str
    polarity: str
    strength: str
    exchange_id: str


@dataclass
class GuardResult:
    evaluation: CompetencyEvaluation
    flags: List[str] = field(default_factory=list)


def apply_guards(ev: CompetencyEvaluation, evidence: List[EvidenceView], *, min_evidence: int,
                 open_contradiction: bool) -> GuardResult:
    flags: List[str] = []
    by_ref: Dict[str, EvidenceView] = {e.ref: e for e in evidence}
    valid_refs = [r for r in ev.evidence_refs if r in by_ref]
    if len(valid_refs) != len(ev.evidence_refs):
        flags.append("invalid_refs_removed")
    ev.evidence_refs = valid_refs
    non_neutral = [e for e in evidence if e.polarity != "neutral"]

    # 1) too little evidence => not sufficiently tested (never a low score)
    if len(non_neutral) < min_evidence and ev.evidence_state != "not_sufficiently_tested":
        if not (len(non_neutral) >= 1 and any(e.strength == "strong" for e in non_neutral) and min_evidence <= 2):
            flags.append("insufficient_evidence_gate")
            ev.evidence_state = "not_sufficiently_tested"
    if ev.evidence_state == "not_sufficiently_tested":
        ev.score = None
        return GuardResult(ev, flags)

    # 2) a score needs at least one valid cited ref
    if ev.score is None or not ev.evidence_refs:
        flags.append("score_without_refs")
        ev.evidence_state = "not_sufficiently_tested"
        ev.score = None
        return GuardResult(ev, flags)

    cited = [by_ref[r] for r in ev.evidence_refs]
    # 3) high scores need strong positive evidence
    if ev.score >= 7 and not any(c.polarity == "positive" and c.strength == "strong" for c in cited):
        flags.append("cap_applied_no_strong_positive")
        ev.score = 6
    # 4) low scores need negative or weak evidence; absence is not weakness
    if ev.score <= 3 and not any(c.polarity == "negative" or c.strength == "weak" for c in cited):
        flags.append("low_score_without_negative_evidence")
        ev.score = None
        ev.evidence_state = "not_sufficiently_tested"
        return GuardResult(ev, flags)
    # 5) polarity sanity: mostly-negative evidence cannot produce a strong score, and vice versa
    pos = sum(1 for c in cited if c.polarity == "positive")
    neg = sum(1 for c in cited if c.polarity == "negative")
    if ev.score >= 7 and neg > pos:
        flags.append("polarity_mismatch_capped")
        ev.score = 5
    if ev.score <= 4 and pos > neg and pos >= 2 and neg == 0:
        flags.append("polarity_mismatch_raised")
        ev.score = 5
    # 6) contradictions that were never resolved
    if open_contradiction and ev.evidence_state != "contradictory":
        flags.append("open_contradiction")
        ev.evidence_state = "contradictory"
    if ev.evidence_state != "contradictory":
        ev.evidence_state = state_for(ev.score)  # band from code, not from the model's label
    return GuardResult(ev, flags)


def confidence(*, exchanges_testing: int, evidence: List[EvidenceView], state: str, ended_early: bool,
               contradictions: int) -> tuple[str, dict]:
    basis = {"exchanges_testing": exchanges_testing, "evidence_items": len(evidence),
             "strong_items": sum(1 for e in evidence if e.strength == "strong"),
             "contradictions": contradictions, "ended_early": ended_early}
    if state == "not_sufficiently_tested":
        return "low", basis
    pos = sum(1 for e in evidence if e.polarity == "positive")
    neg = sum(1 for e in evidence if e.polarity == "negative")
    agreement = max(pos, neg) / max(1, pos + neg)
    basis["agreement"] = round(agreement, 2)
    distinct_exchanges = len({e.exchange_id for e in evidence})
    basis["distinct_exchanges_with_evidence"] = distinct_exchanges
    if distinct_exchanges >= 2 and len(evidence) >= 3 and agreement >= 0.7 and not contradictions:
        level = "high"
    elif len(evidence) >= 2 and agreement >= 0.6:
        level = "moderate"
    else:
        level = "low"
    if ended_early and level == "high":
        level = "moderate"
    return level, basis
