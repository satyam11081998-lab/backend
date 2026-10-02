"""Stage F — decision policy (pure) and the claims/contradiction ledger, unit-level."""

import copy

from interview_intelligence.interview_engine import policy as P
from interview_intelligence.interview_engine.state import new_state
from interview_intelligence.interview_memory import claims as CL


def bp(difficulty="medium", max_probes=2, pushback=0.0, duration=45):
    def item(qid, comps, origin="generated", claim=None, diff=3):
        return {"qid": qid, "competency_ids": comps, "origin": origin, "difficulty": diff, "text": f"Question {qid}?",
                "probe_tree": ["Probe one?", "Probe two?", "Probe three?"],
                "selection_reason": {"claim_ids": [claim] if claim else []}}
    return {
        "config": {"mode": "mixed", "difficulty": difficulty, "duration_minutes": duration},
        "difficulty_vector": {"level": 3, "max_probes": max_probes, "pushback_rate": pushback},
        "competencies": [{"competency_id": "ownership", "importance": "critical", "min_evidence": 2},
                         {"competency_id": "valuation", "importance": "high", "min_evidence": 2},
                         {"competency_id": "communication", "importance": "medium", "min_evidence": 1}],
        "claims_to_investigate": [{"claim_id": "C1", "text": "Led a team of 12", "slots": {"team_size": 12}}],
        "sections": [
            {"id": "intro", "budget_s": 90, "items": [item("Q1", ["communication"], "fixed")], "reserve": []},
            {"id": "cv", "budget_s": 600, "items": [item("Q2", ["ownership"], "cv_specific", "C1")], "reserve": []},
            {"id": "functional", "budget_s": 900, "items": [item("Q3", ["valuation"]), item("Q4", ["valuation"], diff=5),
                                                            item("Q5", ["valuation"], diff=1)], "reserve": []},
            {"id": "closing", "budget_s": 150, "items": [item("Q6", [], "fixed")], "reserve": []},
        ],
    }


def st(b, section="functional", qid="Q3", probes=0):
    s = new_state(b)
    s["section_id"] = section
    s["section_idx"] = [x["id"] for x in b["sections"]].index(section)
    for sid in s["queues"]:
        if qid in s["queues"][sid]:
            s["queues"][sid].remove(qid)
    s["current"] = {"exchange_id": "x", "qid": qid, "probes": probes, "challenged": False, "qualities": []}
    return s


def a(quality="adequate", gaps=("baseline",), focus="quantification"):
    return {"answer_quality": quality, "gaps": list(gaps), "probe_focus": focus}


def test_control_intents():
    b = bp()
    s = st(b)
    for intent, typ in [("end_request", "END_EARLY"), ("break_request", "PAUSE"), ("repeat_request", "REPEAT"),
                        ("thinking_pause", "WAIT"), ("clarification_request", "CLARIFY_QUESTION"),
                        ("meta_question", "REDIRECT"), ("off_topic_question", "REDIRECT")]:
        assert P.decide(s, b, intent=intent, analysis=None, session_key="k").type == typ, intent


def test_refusal_moves_on_and_marks_declined():
    b = bp()
    act = P.decide(st(b), b, intent="refusal", analysis=None, session_key="k")
    assert act.type == "ASK" and act.declined and act.preface == "ack_refusal"


def test_non_answer_nudges_once_then_moves_on():
    b = bp()
    s = st(b)
    s["consecutive"]["non_answers"] = 1
    assert P.decide(s, b, intent="non_answer", analysis=None, session_key="k").type == "NUDGE"
    s["consecutive"]["non_answers"] = 2
    assert P.decide(s, b, intent="non_answer", analysis=None, session_key="k").type == "ASK"


def test_probe_until_limit_then_advance():
    b = bp(max_probes=2)
    assert P.decide(st(b, probes=0), b, intent="answer", analysis=a(), session_key="k").type == "PROBE"
    assert P.decide(st(b, probes=1), b, intent="answer", analysis=a(), session_key="k").type == "PROBE"
    nxt = P.decide(st(b, probes=2), b, intent="answer", analysis=a(), session_key="k")
    assert nxt.type == "ASK" and nxt.qid in ("Q4", "Q5"), "no question loops: never re-asks Q3"


def test_strong_answer_moves_on_and_easy_mode_does_not_probe_adequate():
    b = bp()
    assert P.decide(st(b), b, intent="answer", analysis=a("strong", [], "none"), session_key="k").type == "ASK"
    b_easy = bp(difficulty="easy")
    assert P.decide(st(b_easy), b_easy, intent="answer", analysis=a("adequate"), session_key="k").type == "ASK"
    assert P.decide(st(b_easy), b_easy, intent="answer", analysis=a("weak"), session_key="k").type == "PROBE"


def test_grill_probes_adequate_answers_even_without_focus():
    b = bp(difficulty="grill", max_probes=4)
    act = P.decide(st(b), b, intent="answer", analysis=a("adequate", ["baseline"], "none"), session_key="k")
    assert act.type == "PROBE" and act.focus == "depth"


def test_cv_ladder_progression():
    b = bp(max_probes=2)
    s = st(b, section="cv", qid="Q2")
    act = P.decide(s, b, intent="answer", analysis=a("adequate"), session_key="k")
    assert act.type == "PROBE" and act.probe_text == "Probe one?"
    s["claims"]["C1"]["ladder_step"] = 1
    act = P.decide(s, b, intent="answer", analysis=a("adequate"), session_key="k")
    assert act.probe_text == "Probe two?"


def test_time_up_goes_to_closing_then_final():
    b = bp(duration=15)
    s = st(b)
    s["clock"]["active_s"] = 15 * 60 - 100
    act = P.decide(s, b, intent="answer", analysis=a(), session_key="k")
    assert act.type == "CLOSE_INVITE"
    s["section_id"] = "closing"
    s["closing"]["stage"] = "invited"
    assert P.decide(s, b, intent="answer", analysis=None, session_key="k").type == "CLOSE_FINAL"


def test_skips_items_whose_competency_is_already_sufficient():
    b = bp()
    s = st(b, qid="Q3")
    s["coverage"]["valuation"]["strong"] = 2
    s["coverage"]["valuation"]["asked"] = 1
    act = P.decide(s, b, intent="answer", analysis=a("strong", [], "none"), session_key="k")
    assert act.type == "CLOSE_INVITE", "remaining valuation items skipped; nothing else planned"


def test_difficulty_adapts():
    b = bp()
    s = st(b)
    s["consecutive"]["strong"] = 1
    act = P.decide(s, b, intent="answer", analysis=a("strong", [], "none"), session_key="k")
    assert act.difficulty_delta == 1
    s = st(b)
    s["consecutive"]["weak"] = 1
    act = P.decide(s, b, intent="answer", analysis=a("weak", [], "none"), session_key="k")
    assert act.difficulty_delta == -1


def test_pushback_is_deterministic_per_session_turn():
    b = bp(pushback=0.5)
    s = st(b)
    s["current"]["probes"] = 2  # gap probes exhausted
    results = {P.decide(copy.deepcopy(s), b, intent="answer", analysis=a("adequate"), session_key=f"s{i}").type
               for i in range(30)}
    assert "CHALLENGE" in results and "ASK" in results
    one = [P.decide(copy.deepcopy(s), b, intent="answer", analysis=a("adequate"), session_key="fixed").type for _ in range(5)]
    assert len(set(one)) == 1


def test_contradiction_anchoring_and_once_only():
    b = bp()
    s = new_state(b)
    # different subject: no contradiction
    CL.register(s, [{"text": "my current team is 4", "slot": "team_size", "value": 4}], exchange_id="x1",
                answer_text="...")
    assert CL.detect(s) == []
    # same subject (question investigating C1): contradiction once
    CL.register(s, [{"text": "it was 5 people", "slot": "team_size", "value": 5}], exchange_id="x2",
                answer_text="...", default_anchor="C1")
    new = CL.detect(s)
    assert len(new) == 1 and new[0]["a"] == "C1"
    CL.register(s, [{"text": "about 5", "slot": "team_size", "value": 5}], exchange_id="x3", answer_text="...",
                default_anchor="C1")
    assert CL.detect(s) == [], "same subject+slot is clarified once, not re-raised"
    assert "team size" in CL.describe(new[0], s["memory"])


def test_small_differences_and_self_corrections_are_not_contradictions():
    b = bp()
    s = new_state(b)
    CL.register(s, [{"text": "11 people", "slot": "team_size", "value": 11}], exchange_id="x", answer_text="",
                default_anchor="C1")
    assert CL.detect(s) == [], "12 vs 11 is within tolerance"
    s2 = new_state(b)
    CL.register(s2, [{"text": "growth was 40%", "slot": "growth_pct", "value": 40}], exchange_id="x",
                answer_text="", default_anchor="")
    CL.register(s2, [{"text": "sorry, I meant 20%", "slot": "growth_pct", "value": 20}], exchange_id="x",
                answer_text="Sorry, I meant 20%, not 40.")
    assert CL.detect(s2) == []
