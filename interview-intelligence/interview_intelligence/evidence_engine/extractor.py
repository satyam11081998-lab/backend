"""Evidence Engine (spec §24) — `extract_exchange_evidence` job.

Runs as each question exchange closes, so the post-interview report is fast. Every quote
must be found in the candidate's own words in that exchange (hallucination guard)."""

from __future__ import annotations

import json
import uuid
from typing import Dict, List

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..ai.guard import scan_injection, wrap_untrusted
from ..ai.prompts import EVIDENCE_EXTRACTOR
from ..ai.runner import RunContext, StructuredOutputError, run_structured
from ..db.models import (Claim, EvidenceItem, InterviewBlueprint, InterviewExchange, InterviewMessage, InterviewSession,
                         utcnow)
from ..jobs.queue import register
from ..textutil import containment
from ..versions import ENGINE_VERSIONS
from .metrics import answer_metrics, stuffing_signal
from .schemas import DIMENSIONS, ExchangeEvidence

QUOTE_MIN_MATCH = 0.82

APPLICABLE = {
    "behavioral": ["relevance", "structure", "clarity", "conciseness", "specificity", "ownership", "evidence",
                   "quantification", "reflection", "consistency"],
    "cv_deep_dive": ["relevance", "specificity", "ownership", "evidence", "quantification", "consistency", "depth",
                     "clarity"],
    "situational": ["relevance", "structure", "reasoning", "business_relevance", "clarity", "conciseness", "depth"],
    "functional": ["relevance", "reasoning", "depth", "terminology", "business_relevance", "clarity", "structure"],
    "technical": ["relevance", "reasoning", "depth", "terminology", "structure", "clarity"],
    "case": ["structure", "reasoning", "quantification", "business_relevance", "clarity", "conciseness", "depth"],
    "estimation": ["structure", "reasoning", "quantification", "clarity"],
    "motivation": ["relevance", "specificity", "clarity", "conciseness", "consistency"],
    "company": ["relevance", "business_relevance", "specificity", "reasoning", "clarity"],
    "intro": ["structure", "clarity", "conciseness", "relevance"],
    "pressure": ["reasoning", "evidence", "clarity", "consistency"],
}


def applicable_dimensions(question_type: str) -> List[str]:
    return APPLICABLE.get(question_type, ["relevance", "clarity", "specificity", "reasoning"])


def extract(db: Session, ex: InterviewExchange) -> None:
    sess = db.get(InterviewSession, ex.session_id)
    bp = db.execute(select(InterviewBlueprint).where(InterviewBlueprint.session_id == ex.session_id)).scalar_one()
    msgs = db.execute(select(InterviewMessage).where(InterviewMessage.exchange_id == ex.id)
                      .order_by(InterviewMessage.seq)).scalars().all()
    # A clarifying question, "can you repeat that?" or "give me a second" is not an answer: never evidence.
    not_answers = {"clarification_request", "repeat_request", "thinking_pause", "meta_question", "off_topic_question",
                   "break_request", "end_request"}
    cand_msgs = [m for m in msgs if m.role == "candidate" and m.content and m.content != "[skipped]"
                 and (m.meta or {}).get("intent") not in not_answers]
    if not cand_msgs:
        ex.evidence_status = "skipped"
        return
    candidate_text = "\n".join(m.content for m in cand_msgs)
    transcript = "\n".join(
        f"{'INTERVIEWER' if m.role == 'interviewer' else 'CANDIDATE'} [{m.seq}]"
        f"{' (not an answer)' if m.role == 'candidate' and (m.meta or {}).get('intent') in not_answers else ''}: {m.content}"
        for m in msgs)
    model = {c["competency_id"]: c for c in (bp.competency_model or {}).get("competencies", [])}
    cross = (bp.blueprint or {}).get("cross_cutting", [])
    allowed = [c for c in (ex.competency_ids or []) if c in model] + [c for c in cross if c in model]
    allowed = list(dict.fromkeys(allowed)) or list(model)[:3]
    rubric_view = {cid: {k: (bp.rubric.get(cid) or {}).get(k) for k in
                         ("what_good_looks_like", "strong_signals", "weak_signals", "red_flags")}
                   for cid in allowed}
    qmeta = ex.question_meta or {}
    qtype = qmeta.get("question_type") or "behavioral"
    dims = applicable_dimensions(qtype)
    claim_ids = (qmeta.get("selection_reason") or {}).get("claim_ids") or []
    cv_claims = []
    if claim_ids:
        rows = db.execute(select(Claim).where(Claim.session_id == ex.session_id, Claim.claim_key.in_(claim_ids))).scalars()
        cv_claims = [{"id": c.claim_key, "text": c.text} for c in rows]
    terms = [k.get("term") for k in ((bp.role_profile or {}).get("keywords") or []) if k.get("term")]
    stuffing = stuffing_signal(candidate_text, terms)
    per_answer = [answer_metrics(m.content, duration_ms=(m.meta or {}).get("answer_ms")) for m in cand_msgs]
    ex.metrics = {"answers": per_answer, "stuffing": stuffing}

    body = (
        f"QUESTION: {ex.question_text}\nTYPE: {qtype}\nINTENT: {qmeta.get('intent', '')}\n"
        f"EXPECTED EVIDENCE: {json.dumps(qmeta.get('expected_evidence', []), ensure_ascii=False)}\n"
        f"COMPETENCIES YOU MAY TAG (ids): {json.dumps(allowed)}\n"
        f"RUBRIC EXCERPTS:\n{json.dumps(rubric_view, ensure_ascii=False)}\n"
        f"APPLICABLE DIMENSIONS: {json.dumps(dims)}\n"
        + (f"CV CLAIMS UNDER TEST:\n{wrap_untrusted('cv_claims', 'cv', json.dumps(cv_claims, ensure_ascii=False))}\n"
           if cv_claims else "")
        + ("NOTE: high domain-term density with little substance was measured in this answer — verify application, "
           "do not reward terminology.\n" if stuffing.get("flag") else "")
        + "EXCHANGE TRANSCRIPT:\n" + wrap_untrusted("transcript", str(ex.id), transcript)
    )
    ctx = RunContext(user_id=sess.user_id, session_id=sess.id)

    def _validate(ev: ExchangeEvidence) -> List[str]:
        bad = [i for i, it in enumerate(ev.items) if containment(it.quote, candidate_text) < QUOTE_MIN_MATCH]
        if len(bad) > max(1, len(ev.items) // 2):
            return [f"items {bad} quote text that does not appear in the CANDIDATE's words; quote verbatim"]
        return []

    result = run_structured(EVIDENCE_EXTRACTOR, body, ExchangeEvidence, ctx, validate=_validate,
                            sim_input={"candidate_text": candidate_text, "allowed": allowed, "dims": dims,
                                       "question": ex.question_text, "qtype": qtype,
                                       "probed": sum(1 for m in msgs if m.role == "interviewer") > 1})
    db.execute(delete(EvidenceItem).where(EvidenceItem.exchange_id == ex.id))
    kept = 0
    manipulation_dropped = 0
    for i, it in enumerate(result.items):
        if it.competency_id not in model:
            continue
        verified = containment(it.quote, candidate_text) >= QUOTE_MIN_MATCH
        if not verified:
            continue  # unverifiable quote = not evidence
        if scan_injection(it.quote):
            manipulation_dropped += 1  # an instruction aimed at the system is never evidence of competence
            continue
        kept += 1
        db.add(EvidenceItem(
            session_id=ex.session_id, exchange_id=ex.id, ref=f"E{ex.seq}.{kept}", competency_id=it.competency_id,
            sub_competency=it.sub_competency[:160], evidence_type=it.type, polarity=it.polarity, strength=it.strength,
            quote=it.quote[:1000], quote_verified=True, interpretation=it.interpretation[:1000],
            ownership=it.ownership, cv_consistency=it.cv_consistency, confidence=float(it.confidence),
            extractor_version=ENGINE_VERSIONS["evidence"]))
    analysis = result.model_dump()
    analysis["items"] = None  # stored as rows
    analysis["dimensions"] = {k: v for k, v in (analysis.get("dimensions") or {}).items() if k in DIMENSIONS and k in dims}
    analysis["dropped_manipulation_quotes"] = manipulation_dropped
    analysis["dropped_unverified_quotes"] = len(result.items) - kept - manipulation_dropped
    ex.analysis = analysis
    ex.evidence_status = "ready"
    # CV claim verification status from evidence (supported / unsupported / contradicted)
    for cid in claim_ids:
        cl = db.execute(select(Claim).where(Claim.session_id == ex.session_id, Claim.claim_key == cid)).scalar_one_or_none()
        if cl is None:
            continue
        pols = [i for i in result.items if i.cv_consistency != "n/a" or i.type in ("ownership", "outcome", "quantification")]
        if any(i.cv_consistency == "inconsistent" for i in result.items):
            cl.verification_status = "contradicted"
        elif any(i.polarity == "positive" and i.strength == "strong" for i in pols):
            cl.verification_status = "supported"
        elif any(i.polarity == "negative" for i in pols):
            cl.verification_status = "unsupported"
        else:
            cl.verification_status = "partially_supported"


def _on_dead(db: Session, payload: dict, err: str) -> None:
    ex = db.get(InterviewExchange, uuid.UUID(payload["exchange_id"]))
    if ex is not None:
        ex.evidence_status = "failed"


@register("extract_exchange_evidence", on_dead=_on_dead)
def handle(db: Session, payload: dict) -> None:
    ex = db.execute(select(InterviewExchange).where(InterviewExchange.id == uuid.UUID(payload["exchange_id"]))
                    .with_for_update()).scalar_one_or_none()
    if ex is None or ex.evidence_status == "ready":
        return
    extract(db, ex)
