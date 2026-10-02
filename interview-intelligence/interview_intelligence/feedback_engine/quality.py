"""Bad-feedback detector (spec §75–§76). Deterministic; runs before the LLM QA pass."""

from __future__ import annotations

import re
from typing import Dict, List, Set

from ..evaluation_engine.evaluator import PROTECTED_RX

GENERIC = re.compile(
    r"\b(work on your communication|improve (your )?communication( skills)?|be more confident|"
    r"(gain|build|show) (more )?confidence|study more|read more|practice more|use (the )?star( method| framework)?|"
    r"improve (your )?articulation|be more (concise|specific|structured)|work on (your )?(soft skills|presentation))\b",
    re.IGNORECASE,
)
HIRING_PREDICTION = re.compile(r"\b(you (will|would) (get|be) (hired|selected|rejected)|hire probability|"
                               r"chance(s)? of (selection|getting the job)|likely to be (hired|selected|rejected)|"
                               r"\d{1,3}\s?% (chance|likely|probability|fit))\b", re.IGNORECASE)
NUMBER = re.compile(r"(?<![A-Za-z#.])(\d+(?:\.\d+)?)(?:\s?%|\s?(?:seconds?|secs?|minutes?|words?|times?))?")
REF_X = re.compile(r"^X\d+$")


def _numbers(text: str) -> Set[str]:
    return {m.group(1) for m in NUMBER.finditer(text or "")}


def _wc(s: str) -> int:
    return len((s or "").split())


def check_development(area: dict, *, exchange_refs: Set[str], metrics: Dict[str, float],
                      assessments: Dict[str, dict]) -> List[str]:
    p: List[str] = []
    text_all = " ".join(str(area.get(k) or "") for k in ("title", "observed_problem", "why_it_matters", "what_to_do",
                                                          "practice"))
    if GENERIC.search(text_all) and _wc(area.get("observed_problem", "")) < 15:
        p.append("generic advice without a specific observed problem")
    refs = [r for r in area.get("example_refs") or [] if r in exchange_refs]
    if not refs:
        p.append("no valid example reference (X#) to where it happened")
    if _wc(area.get("observed_problem", "")) < 10:
        p.append("observed_problem is too thin to act on")
    if _wc(area.get("what_to_do", "")) < 8:
        p.append("what_to_do is missing or too short")
    if _wc(area.get("practice", "")) < 8:
        p.append("practice drill is missing or too short")
    # numbers presented as observations must be measured (cited) or quoted from the candidate
    allowed: Set[str] = set()
    for mid in area.get("measured_basis") or []:
        if mid in metrics:
            v = metrics[mid]
            allowed |= {str(v), str(int(v)) if float(v).is_integer() else str(v), str(round(float(v)))}
    allowed |= _numbers(area.get("example_quote", ""))
    stray = _numbers(area.get("observed_problem", "")) - allowed
    if stray:
        p.append(f"states unmeasured numbers {sorted(stray)} as observations")
    if PROTECTED_RX.search(text_all):
        p.append("mentions a non-job-relevant attribute (accent, grammar, personality, protected characteristic)")
    comps = [c for c in area.get("competency_ids") or [] if c in assessments]
    if comps:
        states = [assessments[c].get("evidence_state") for c in comps]
        if all(s == "not_sufficiently_tested" for s in states):
            p.append("treats an untested competency as a weakness")
        if all(s == "strong" for s in states):
            p.append("contradicts the assessment (competency assessed as strong)")
    return p


def check_executive(text: str) -> List[str]:
    p = []
    if _wc(text) < 25:
        p.append("executive assessment too short")
    if HIRING_PREDICTION.search(text or ""):
        p.append("makes a hiring prediction")
    if PROTECTED_RX.search(text or ""):
        p.append("mentions a non-job-relevant attribute")
    return p


def valid_refs(refs: List[str], allowed: Set[str]) -> List[str]:
    return [r for r in refs or [] if r in allowed]
