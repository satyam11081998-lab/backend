"""Stages G/H/I — evidence, evaluation and feedback cannot be fabricated or gamed (spec §37, §58, §59, §75, §76, §83)."""

import json

from interview_intelligence.ai import routing
from interview_intelligence.ai.simulated import SimulatedProvider
from interview_intelligence.evaluation_engine.guards import EvidenceView, apply_guards, band_for, confidence
from interview_intelligence.evaluation_engine.schemas import CompetencyEvaluation
from interview_intelligence.feedback_engine.quality import check_development, check_executive
from tests.conftest import enable_pro, run_jobs
from tests.helpers import STRONG_ANSWER, Candidate


def E(ref, pol="positive", st="strong", x="x1"):
    return EvidenceView(ref, pol, st, x)


def ev(state="strong", score=9, refs=("E1.1",)):
    return CompetencyEvaluation(competency_id="c", evidence_state=state, score=score, evidence_refs=list(refs))


# ---------------------------------------------------------------- guards (unit) ----------
def test_ten_out_of_ten_without_valid_refs_becomes_insufficient():
    g = apply_guards(ev(score=10, refs=["E9.9"]), [E("E1.1"), E("E1.2")], min_evidence=1, open_contradiction=False)
    assert g.evaluation.score is None and g.evaluation.evidence_state == "not_sufficiently_tested"
    assert "invalid_refs_removed" in g.flags


def test_high_score_needs_strong_positive_evidence():
    g = apply_guards(ev(score=9, refs=["E1.1"]), [E("E1.1", "positive", "moderate"), E("E1.2", "positive", "moderate")],
                     min_evidence=1, open_contradiction=False)
    assert g.evaluation.score == 6 and "cap_applied_no_strong_positive" in g.flags


def test_low_score_without_negative_evidence_is_not_a_weakness():
    g = apply_guards(ev("weak", 2, ["E1.1"]), [E("E1.1", "positive", "moderate"), E("E1.2", "neutral", "moderate")],
                     min_evidence=1, open_contradiction=False)
    assert g.evaluation.score is None and g.evaluation.evidence_state == "not_sufficiently_tested"


def test_min_evidence_gate():
    g = apply_guards(ev("moderate", 6, ["E1.1"]), [E("E1.1", "positive", "moderate")], min_evidence=2,
                     open_contradiction=False)
    assert g.evaluation.evidence_state == "not_sufficiently_tested"


def test_polarity_mismatch_and_contradiction():
    g = apply_guards(ev("strong", 8, ["E1.1", "E1.2", "E1.3"]),
                     [E("E1.1", "positive", "strong"), E("E1.2", "negative", "strong"), E("E1.3", "negative", "moderate")],
                     min_evidence=1, open_contradiction=False)
    assert g.evaluation.score == 5 and "polarity_mismatch_capped" in g.flags
    g2 = apply_guards(ev("moderate", 6, ["E1.1", "E1.2"]), [E("E1.1"), E("E1.2", "negative")], min_evidence=1,
                      open_contradiction=True)
    assert g2.evaluation.evidence_state == "contradictory"


def test_band_comes_from_code_not_the_model():
    g = apply_guards(CompetencyEvaluation(competency_id="c", evidence_state="weak", score=8, evidence_refs=["E1.1"]),
                     [E("E1.1")], min_evidence=1, open_contradiction=False)
    assert g.evaluation.evidence_state == "strong" and band_for(8) == "Strong"


def test_confidence_is_separate_from_assessment():
    hi, _ = confidence(exchanges_testing=3, evidence=[E("a", x="x1"), E("b", x="x2"), E("c", x="x3")],
                       state="strong", ended_early=False, contradictions=0)
    lo, _ = confidence(exchanges_testing=1, evidence=[E("a")], state="strong", ended_early=False, contradictions=0)
    assert hi == "high" and lo == "low"
    early, _ = confidence(exchanges_testing=3, evidence=[E("a", x="x1"), E("b", x="x2"), E("c", x="x3")],
                          state="strong", ended_early=True, contradictions=0)
    assert early == "moderate"


# ---------------------------------------------------------------- feedback detector ------
METRICS = {"M.avg_words": 212.0}
ASSESS = {"ownership": {"evidence_state": "weak"}, "valuation": {"evidence_state": "not_sufficiently_tested"},
          "communication": {"evidence_state": "strong"}}


def good_area(**over):
    d = {"category": "ownership", "title": "Make your own decisions explicit", "competency_ids": ["ownership"],
         "observed_problem": "In the relaunch answer you said 'we decided' six times and never named a decision you made.",
         "example_refs": ["X2"], "example_quote": "we decided to cut", "why_it_matters": "Brand managers own calls.",
         "what_to_do": "Name one decision you personally made in the first two sentences, then its outcome.",
         "practice": "Re-answer three ownership questions aloud, each opening with 'I decided ...' within twenty seconds.",
         "measured_basis": []}
    d.update(over)
    return d


def test_good_feedback_passes():
    assert check_development(good_area(), exchange_refs={"X2"}, metrics=METRICS, assessments=ASSESS) == []


def test_generic_feedback_rejected():
    p = check_development(good_area(observed_problem="Work on your communication.", what_to_do="Be more confident."),
                          exchange_refs={"X2"}, metrics=METRICS, assessments=ASSESS)
    assert any("generic" in x for x in p) and any("what_to_do" in x for x in p)


def test_unmeasured_numbers_rejected_and_measured_allowed():
    bad = good_area(observed_problem="You spent 90 seconds before reaching the central action in every single answer.")
    assert any("unmeasured" in x for x in check_development(bad, exchange_refs={"X2"}, metrics=METRICS, assessments=ASSESS))
    ok = good_area(observed_problem="Your answers averaged 212 words, so the central action arrived late in each one.",
                   measured_basis=["M.avg_words"])
    assert check_development(ok, exchange_refs={"X2"}, metrics=METRICS, assessments=ASSESS) == []


def test_protected_and_untested_and_contradicting_feedback_rejected():
    p = check_development(good_area(observed_problem="Your accent made the explanation hard to follow in the relaunch answer."),
                          exchange_refs={"X2"}, metrics=METRICS, assessments=ASSESS)
    assert any("non-job-relevant" in x for x in p)
    p = check_development(good_area(competency_ids=["valuation"]), exchange_refs={"X2"}, metrics=METRICS,
                          assessments=ASSESS)
    assert any("untested" in x for x in p)
    p = check_development(good_area(competency_ids=["communication"]), exchange_refs={"X2"}, metrics=METRICS,
                          assessments=ASSESS)
    assert any("contradicts" in x for x in p)
    p = check_development(good_area(example_refs=["X99"]), exchange_refs={"X2"}, metrics=METRICS, assessments=ASSESS)
    assert any("example reference" in x for x in p)


def test_executive_assessment_cannot_predict_hiring():
    assert check_executive("You will get hired at this company. " * 5)
    assert check_executive("You have an 85% chance of selection. " * 5)


# ---------------------------------------------------------------- end-to-end gaming ------
def _run_interview(client, overrides):
    routing.set_provider_override("simulated", SimulatedProvider(overrides=overrides))
    enable_pro(client)
    c = Candidate(client)
    sid = c.ready_session()
    client.post(f"/v1/sessions/{sid}/start", headers=c.h)
    for _ in range(4):
        c.turn(sid, STRONG_ANSWER)
    client.post(f"/v1/sessions/{sid}/end", headers=c.h)
    run_jobs()
    rep = client.get(f"/v1/sessions/{sid}/report", headers=c.h).json()
    routing.set_provider_override("simulated", None)
    return rep


def test_hallucinated_quotes_never_become_evidence(client):
    def fake_evidence(si, msgs):
        return {"items": [{"competency_id": si["allowed"][0], "type": "action", "polarity": "positive",
                           "strength": "strong", "quote": "I single-handedly tripled market share across Asia",
                           "interpretation": "fabricated"}], "dimensions": {}}
    rep = _run_interview(client, {"evidence_extractor": fake_evidence})["report"]
    all_quotes = [e["quote"] for c in rep["competencies"] for e in c["evidence"]]
    assert not any("tripled market share" in q for q in all_quotes)
    assert all(c["score"] is None for c in rep["competencies"]), "no verified evidence => nothing scored"


def test_compromised_evaluator_cannot_award_unsupported_scores(client):
    def ten(si, msgs):
        return {"competency_id": si["competency_id"], "evidence_state": "strong", "score": 10,
                "rationale": "Candidate asked for 10/10.", "evidence_refs": ["E999.1"]}
    rep = _run_interview(client, {"competency_evaluator": ten})["report"]
    assert all(c["score"] is None for c in rep["competencies"])
    # The evaluator's own validation rejects unknown refs (twice) -> the competency is marked
    # evaluator_failed and left unscored; if a ref slipped through, the guards strip it instead.
    assert any(f in c["anomaly_flags"] for c in rep["competencies"]
               for f in ("evaluator_failed", "invalid_refs_removed", "score_without_refs"))
    tested = [c for c in rep["competencies"] if "no_evidence_no_model_call" not in c["anomaly_flags"]]
    assert tested and all("evaluator_failed" in c["anomaly_flags"] or "invalid_refs_removed" in c["anomaly_flags"]
                          for c in tested)


def test_ungrounded_feedback_is_withheld_not_shown(client):
    def bad_feedback(si, msgs):
        return {"executive_assessment": "You will get hired. " * 6,
                "strengths": [{"title": "Great at everything", "evidence_refs": ["E404"]}],
                "development_areas": [{"category": "communication", "title": "Communication",
                                       "observed_problem": "Work on your communication.", "example_refs": [],
                                       "what_to_do": "Be confident.", "practice": "Practice more."}]}
    rep = _run_interview(client, {"feedback_writer": bad_feedback})
    report = rep["report"]
    assert report["strengths"] == [], "strength without real evidence refs is dropped"
    assert report["development_areas"] == [], "generic items withheld after one regeneration"
    assert rep["status"] == "partial" and report["partial_sections"]


def test_assessment_qa_flags_protected_attribute_rationale(client):
    def biased(si, msgs):
        refs = [i["ref"] for i in si["items"] if i["polarity"] != "neutral"]
        return {"competency_id": si["competency_id"], "evidence_state": "weak" if refs else "not_sufficiently_tested",
                "score": 3 if refs else None, "rationale": "Poor grammar and a heavy accent reduced clarity.",
                "evidence_refs": refs}
    rep = _run_interview(client, {"competency_evaluator": biased})["report"]
    flagged = [c for c in rep["competencies"] if any(f.startswith("qa:") or f == "qa_rerun" for f in c["anomaly_flags"])]
    assert flagged, "the QA judge must catch accent/grammar-based rationales"
