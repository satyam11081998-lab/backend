"""Feedback Engine (spec §39–§47, §75): evidence -> actionable feedback, double-checked."""

from __future__ import annotations

import json
from typing import Dict, List, Optional, Set, Tuple

from ..ai.guard import wrap_untrusted
from ..ai.prompts import FEEDBACK_QA, FEEDBACK_WRITER
from ..ai.runner import RunContext, StructuredOutputError, run_structured
from .quality import check_development, check_executive, valid_refs
from .schemas import FeedbackBundle, FeedbackQA


def _body(ctx_payload: dict, problems: Optional[str] = None) -> str:
    body = (
        f"ROLE: {json.dumps(ctx_payload['role'], ensure_ascii=False)}\n"
        f"INTERVIEW: {json.dumps(ctx_payload['interview'], ensure_ascii=False)}\n"
        f"MEASURED METRICS (cite ids in measured_basis for any number you state):\n"
        f"{json.dumps(ctx_payload['metrics'], ensure_ascii=False)}\n"
        f"COMPETENCY ASSESSMENTS:\n{json.dumps(ctx_payload['assessments'], ensure_ascii=False)}\n"
        f"RECURRING PATTERNS FROM EARLIER INTERVIEWS: {json.dumps(ctx_payload['recurring'], ensure_ascii=False)}\n"
        "QUESTION EXCHANGES (X# refs) WITH CANDIDATE EVIDENCE (E# refs):\n"
        + wrap_untrusted("exchanges", "all", json.dumps(ctx_payload["exchanges"], ensure_ascii=False))
        + "\nCV CLAIMS INVESTIGATED:\n" + wrap_untrusted("claims", "cv", json.dumps(ctx_payload["claims"], ensure_ascii=False))
    )
    if problems:
        body += ("\n\nYOUR PREVIOUS DRAFT WAS REJECTED FOR THESE REASONS — fix them; drop any item you cannot ground:\n"
                 + problems)
    return body


def _sanitize(bundle: FeedbackBundle, *, e_refs: Set[str], x_refs: Set[str], c_refs: Set[str],
              metrics: Dict[str, float], assessments: Dict[str, dict]) -> Tuple[FeedbackBundle, List[Tuple[int, List[str]]], List[str]]:
    any_ref = e_refs | x_refs | c_refs
    bundle.strengths = [s for s in bundle.strengths if valid_refs(s.evidence_refs, e_refs | x_refs)]
    for s in bundle.strengths:
        s.evidence_refs = valid_refs(s.evidence_refs, e_refs | x_refs)
    learned = bundle.interviewer_learned
    for field in ("strong_signals", "weak_signals", "unproven_claims", "missing_evidence", "potential_concerns"):
        items = getattr(learned, field)
        kept = []
        for it in items:
            it.refs = valid_refs(it.refs, any_ref)
            # missing evidence may legitimately have no ref (nothing was said); everything else must be grounded
            if it.refs or field == "missing_evidence":
                kept.append(it)
        setattr(learned, field, kept)
    failures: List[Tuple[int, List[str]]] = []
    for i, d in enumerate(bundle.development_areas):
        d.example_refs = valid_refs(d.example_refs, x_refs)
        d.measured_basis = [m for m in d.measured_basis if m in metrics]
        probs = check_development(d.model_dump(), exchange_refs=x_refs, metrics=metrics, assessments=assessments)
        if probs:
            failures.append((i, probs))
    exec_problems = check_executive(bundle.executive_assessment)
    n = len(bundle.development_areas)
    for item in bundle.preparation_plan.items:
        item.linked_development = [x for x in item.linked_development if 0 <= x < n]
    return bundle, failures, exec_problems


def write_feedback(ctx: RunContext, payload: dict, *, e_refs: Set[str], x_refs: Set[str], c_refs: Set[str],
                   metrics: Dict[str, float], assessments: Dict[str, dict]) -> Tuple[Optional[FeedbackBundle], dict]:
    """Returns (bundle or None, qa_report). Generic or ungrounded items never reach the candidate."""
    report = {"attempts": 0, "dropped": [], "regenerated": False, "llm_qa": None, "exec_problems": []}
    problems_text = None
    bundle: Optional[FeedbackBundle] = None
    for attempt in (1, 2):
        report["attempts"] = attempt
        try:
            bundle = run_structured(FEEDBACK_WRITER, _body(payload, problems_text), FeedbackBundle, ctx,
                                    sim_input=payload)
        except StructuredOutputError:
            return None, report
        bundle, failures, exec_problems = _sanitize(bundle, e_refs=e_refs, x_refs=x_refs, c_refs=c_refs,
                                                    metrics=metrics, assessments=assessments)
        llm_fail: Dict[int, List[str]] = {}
        try:
            qa = run_structured(FEEDBACK_QA, "FEEDBACK:\n" + json.dumps({
                "executive_assessment": bundle.executive_assessment,
                "development_areas": [d.model_dump() for d in bundle.development_areas]}, ensure_ascii=False)
                + "\nASSESSMENTS:\n" + json.dumps(payload["assessments"], ensure_ascii=False)
                + "\nMEASURED METRICS:\n" + json.dumps(payload["metrics"], ensure_ascii=False),
                FeedbackQA, ctx, sim_input={"n": len(bundle.development_areas)})
            report["llm_qa"] = qa.model_dump()
            for it in qa.development_areas:
                if not it.passed and 0 <= it.index < len(bundle.development_areas):
                    llm_fail[it.index] = it.problems or ["failed QA"]
            if not qa.executive_assessment_ok:
                exec_problems += qa.executive_problems or ["executive assessment failed QA"]
        except StructuredOutputError:
            report["llm_qa"] = {"unavailable": True}
        all_fail: Dict[int, List[str]] = {i: p for i, p in failures}
        for i, p in llm_fail.items():
            all_fail.setdefault(i, []).extend(p)
        if not all_fail and not exec_problems:
            break
        if attempt == 1:
            report["regenerated"] = True
            lines = [f"- development_areas[{i}] ({bundle.development_areas[i].title}): {'; '.join(p)}"
                     for i, p in all_fail.items()]
            lines += [f"- executive_assessment: {p}" for p in exec_problems]
            problems_text = "\n".join(lines)
            continue
        # second attempt still failing: drop the failing items, keep the rest, mark partial
        keep = [d for i, d in enumerate(bundle.development_areas) if i not in all_fail]
        report["dropped"] = [{"title": bundle.development_areas[i].title, "problems": p} for i, p in all_fail.items()]
        remap = {}
        j = 0
        for i in range(len(bundle.development_areas)):
            if i not in all_fail:
                remap[i] = j
                j += 1
        bundle.development_areas = keep
        for item in bundle.preparation_plan.items:
            item.linked_development = [remap[x] for x in item.linked_development if x in remap]
        report["exec_problems"] = exec_problems
    return bundle, report
