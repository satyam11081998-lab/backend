"""Evaluator (spec §66–§68): one competency at a time, rubric fixed in advance, evidence
refs only, then deterministic gates + confidence, then an independent QA judge."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

from ..ai.guard import wrap_untrusted
from ..ai.prompts import ASSESSMENT_QA, COMPETENCY_EVALUATOR
from ..ai.runner import RunContext, StructuredOutputError, run_structured
from ..db.models import EvidenceItem
from .guards import EvidenceView, GuardResult, apply_guards, band_for, confidence
from .schemas import AssessmentQA, CompetencyEvaluation

CANONICAL_BANDS = ("9-10 Exceptional | 7-8 Strong | 5-6 Moderate | 3-4 Below expected | 1-2 Needs development | "
                   "not_sufficiently_tested when evidence is too thin")


def evaluate_competency(*, ctx: RunContext, comp: dict, rubric: dict, items: List[EvidenceItem],
                        exchanges_testing: int, flags: List[str], qa_feedback: str = "") -> CompetencyEvaluation:
    ev_payload = [{"ref": e.ref, "type": e.evidence_type, "polarity": e.polarity, "strength": e.strength,
                   "ownership": e.ownership, "quote": e.quote, "interpretation": e.interpretation}
                  for e in items]
    body = (
        f"COMPETENCY: {comp['competency_id']} — {comp.get('name')} (importance: {comp.get('importance')})\n"
        f"CANONICAL BANDS: {CANONICAL_BANDS}\n"
        f"ROLE-SPECIFIC RUBRIC (fixed before the interview):\n{json.dumps(rubric, ensure_ascii=False)}\n"
        f"COVERAGE: {exchanges_testing} question exchange(s) targeted this competency; "
        f"minimum evidence for a judgement: {comp.get('min_evidence', 1)} item(s).\n"
        f"FLAGS: {', '.join(flags) or 'none'}\n"
        "EVIDENCE (candidate's own words, verified quotes):\n"
        + wrap_untrusted("evidence", comp["competency_id"], json.dumps(ev_payload, ensure_ascii=False))
    )
    if qa_feedback:
        body += f"\n\nAN AUDITOR FLAGGED YOUR PREVIOUS ASSESSMENT: {qa_feedback}\nRe-assess strictly from the evidence."
    refs = {e.ref for e in items}

    def _validate(ev: CompetencyEvaluation) -> List[str]:
        p = []
        if ev.competency_id != comp["competency_id"]:
            ev.competency_id = comp["competency_id"]
        unknown = [r for r in ev.evidence_refs if r not in refs]
        if unknown:
            p.append(f"evidence_refs not in the supplied evidence: {unknown}")
        if ev.evidence_state != "not_sufficiently_tested" and ev.score is None:
            p.append("score is required unless evidence_state is not_sufficiently_tested")
        return p

    return run_structured(COMPETENCY_EVALUATOR, body, CompetencyEvaluation, ctx, validate=_validate,
                          sim_input={"competency_id": comp["competency_id"], "items": ev_payload})


def assess(*, ctx: RunContext, comp: dict, rubric: dict, items: List[EvidenceItem], exchanges_testing: int,
           ended_early: bool, open_contradiction: bool, contradictions: int, flags: List[str],
           qa_feedback: str = "") -> Tuple[CompetencyEvaluation, List[str], str, dict]:
    views = [EvidenceView(e.ref, e.polarity, e.strength, str(e.exchange_id)) for e in items]
    if not items:
        ev = CompetencyEvaluation(competency_id=comp["competency_id"], evidence_state="not_sufficiently_tested",
                                  missing_evidence=["Not tested in this interview" if exchanges_testing == 0 else
                                                    "Answers did not provide assessable evidence"])
        conf, basis = confidence(exchanges_testing=exchanges_testing, evidence=[], state=ev.evidence_state,
                                 ended_early=ended_early, contradictions=contradictions)
        return ev, ["no_evidence_no_model_call"], conf, basis
    try:
        ev = evaluate_competency(ctx=ctx, comp=comp, rubric=rubric, items=items, exchanges_testing=exchanges_testing,
                                 flags=flags, qa_feedback=qa_feedback)
    except StructuredOutputError:
        ev = CompetencyEvaluation(competency_id=comp["competency_id"], evidence_state="not_sufficiently_tested",
                                  rationale="The evaluator was unavailable; no score is shown rather than a guess.")
        conf, basis = confidence(exchanges_testing=exchanges_testing, evidence=views, state=ev.evidence_state,
                                 ended_early=ended_early, contradictions=contradictions)
        return ev, ["evaluator_failed"], conf, basis
    gr: GuardResult = apply_guards(ev, views, min_evidence=int(comp.get("min_evidence", 1)),
                                   open_contradiction=open_contradiction)
    conf, basis = confidence(exchanges_testing=exchanges_testing, evidence=views, state=gr.evaluation.evidence_state,
                             ended_early=ended_early, contradictions=contradictions)
    return gr.evaluation, gr.flags, conf, basis


import re as _re

PROTECTED_RX = _re.compile(
    r"\b(accent|grammar|grammatical|fluency|fluent english|native speaker|non-native|personality|introvert(ed)?|"
    r"extrovert(ed)?|intelligen(t|ce)|iq|nervous|anxious|anxiety|his age|her age|their age|too old|too young|"
    r"gender|religio(n|us)|caste|ethnicity)\b", _re.IGNORECASE)


def deterministic_qa(assessments: List[dict]) -> List[dict]:
    findings = []
    scored = [a for a in assessments if a.get("score") is not None]
    if len(scored) >= 5 and len({a["score"] for a in scored}) == 1:
        findings.append({"competency_id": "", "problem": "other", "severity": "medium",
                         "detail": "All scored competencies received the same score."})
    for a in assessments:
        text = a.get("rationale", "") + " " + " ".join(a.get("gaps", []))
        if PROTECTED_RX.search(text):
            findings.append({"competency_id": a["competency_id"], "problem": "style_bias", "severity": "high",
                             "detail": "Rationale references a non-job-relevant attribute."})
    return findings


def qa_judge(ctx: RunContext, assessments: List[dict]) -> List[dict]:
    view = [{k: a.get(k) for k in ("competency_id", "evidence_state", "score", "rationale", "gaps", "evidence")}
            for a in assessments]
    try:
        qa = run_structured(ASSESSMENT_QA, "ASSESSMENTS:\n" + wrap_untrusted("assessments", "all",
                            json.dumps(view, ensure_ascii=False)), AssessmentQA, ctx, sim_input={"assessments": view})
        llm = [f.model_dump() for f in qa.findings]
    except StructuredOutputError:
        llm = []
    return deterministic_qa(assessments) + llm


def band(score: Optional[int]) -> str:
    return band_for(score)
