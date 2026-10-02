"""The interviewer stays with the conversation (tester reports, 2026-10-02):
 * a clarifying question is answered, never scored as an answer and skipped;
 * a follow-up never refers to something the candidate didn't say ("what was the segment about?");
 * the same question is never asked twice in different words."""

from interview_intelligence.ai import routing
from interview_intelligence.ai.runner import RunContext
from interview_intelligence.ai.simulated import HANDLERS, SimulatedProvider
from interview_intelligence.interview_engine import grounding as G
from interview_intelligence.interview_engine import policy as P
from interview_intelligence.interview_engine.interviewer import speak
from interview_intelligence.interview_engine.policy import Action
from tests.helpers import STRONG_ANSWER, WEAK_ANSWER
from tests.test_interview_adversarial import started
from tests.test_policy_and_memory import bp as policy_bp, st as policy_st

Q = "Tell me about a decision you owned that changed the direction of a product."
A = ("At Infosys I owned the migration of our PKI certificates to a new vendor because renewals kept failing; "
     "I cut outages by 40%.")


def test_ungrounded_references_are_caught():
    assert G.ungrounded("What was the segment about?", [Q, A]) == ["segment about"]
    assert G.ungrounded("How did you size the lapsed-buyer segment?", [Q, A])
    assert G.ungrounded("Earlier you mentioned the pricing change. Why that?", [Q, A])
    assert G.ungrounded("How did the 25% drop in outages hold up?", [Q, A])
    for ok in ("What was the result, and how did you measure it?", "Why did you choose the new vendor?",
               "How did the 40% drop hold up over time?", "What was your personal part in that?",
               "Looking back, what would you do differently?", "Which of the certificates failed most often?",
               "You said renewals kept failing. What was the baseline?"):
        assert G.ungrounded(ok, [Q, A]) == [], ok


def test_same_question_in_other_words():
    assert G.same_question("Tell me about a time you had to influence a stakeholder without authority.",
                           "Describe a situation where you influenced stakeholders without having authority over them.")
    assert not G.same_question("Tell me about a decision you owned.",
                               "How do you prioritise a backlog with competing stakeholder asks?")


def test_planned_follow_up_that_presupposes_details_is_not_used():
    bp = {"role": {"title": "Brand Manager"}, "config": {"duration_minutes": 30}}
    action = Action("PROBE", qid="Q3", focus="quantification", probe_text="How did you size the lapsed-buyer segment?")
    text, used, guard = speak(action, ctx=RunContext(), bp=bp, question_text=Q, last_answer=A, memory_refs=[],
                              recent_openers=[], grounding=[Q, A])
    assert "segment" not in text.lower() and "?" in text
    assert "planned follow-up not used" in guard
    # a grounded planned follow-up is still used
    action = Action("PROBE", qid="Q3", focus="outcome", probe_text="How did you measure the outages before the migration?")
    text, _, guard = speak(action, ctx=RunContext(), bp=bp, question_text=Q, last_answer=A, memory_refs=[],
                           recent_openers=[], grounding=[Q, A])
    assert text == "How did you measure the outages before the migration?" and not guard


def test_follow_ups_rotate_instead_of_repeating():
    bp = {"role": {"title": "x"}, "config": {}}
    action = Action("PROBE", qid="Q3", focus="specificity")
    first, _, _ = speak(action, ctx=RunContext(), bp=bp, question_text=Q, last_answer=A, memory_refs=[],
                        recent_openers=[], grounding=[Q, A])
    second, _, _ = speak(action, ctx=RunContext(), bp=bp, question_text=Q, last_answer=A, memory_refs=[],
                         recent_openers=[], grounding=[Q, A], used_foci=["specificity"])
    assert first != second


def test_hallucinated_follow_up_from_the_model_is_replaced(client):
    def inventive(si, msgs):
        if (si.get("action") or {}).get("type") == "PROBE":
            return "Interesting. What was the segment about, and how big was it?"
        return HANDLERS["interviewer"](si)
    routing.set_provider_override("simulated", SimulatedProvider(overrides={"interviewer": inventive}))
    try:
        c, sid = started(client, difficulty="hard")
        replies = [c.turn(sid, WEAK_ANSWER).json()["messages"][-1] for _ in range(3)]
        assert all("segment" not in r["content"].lower() for r in replies), [r["content"] for r in replies]
        admin = client.get(f"/v1/sessions/{sid}/room", headers=c.h).json()
        assert admin["messages"]
        from interview_intelligence.db.models import InterviewEvent, InterviewMessage
        from interview_intelligence.db.session import db_session
        from sqlalchemy import select
        with db_session() as db:
            notes = [e.payload["note"] for e in db.execute(select(InterviewEvent).where(
                InterviewEvent.type == "line_guard")).scalars()]
            metas = [m.meta for m in db.execute(select(InterviewMessage).where(
                InterviewMessage.role == "interviewer")).scalars()]
        assert any("never said" in n for n in notes), notes
        assert any(m.get("guard") for m in metas)
    finally:
        routing.set_provider_override("simulated", None)


def test_clarifying_question_is_answered_not_scored_and_skipped(client):
    c, sid = started(client)
    r = c.turn(sid, STRONG_ANSWER).json()          # past the intro: now on a real question
    question = r["messages"][-1]["content"]
    from interview_intelligence.db.models import InterviewMessage
    from interview_intelligence.db.session import db_session
    from sqlalchemy import select

    def last_interviewer():
        with db_session() as db:
            m = db.execute(select(InterviewMessage).where(InterviewMessage.role == "interviewer")
                           .order_by(InterviewMessage.seq.desc())).scalars().first()
            return m.exchange_id, m.action
    before, _ = last_interviewer()
    for ask in ("Do you mean in my current role?", "Like for my last job, or a college project?"):
        r = c.turn(sid, ask).json()
        assert r["messages"][-1]["role"] == "interviewer"
        xid, action = last_interviewer()
        assert xid == before and action == "CLARIFY_QUESTION", "same question, clarified — not a new one"
    with db_session() as db:
        last = db.execute(select(InterviewMessage).where(InterviewMessage.role == "interviewer")
                          .order_by(InterviewMessage.seq.desc())).scalars().first()
        cands = db.execute(select(InterviewMessage).where(InterviewMessage.role == "candidate")
                           .order_by(InterviewMessage.seq.desc())).scalars().first()
    assert last.action == "CLARIFY_QUESTION"
    assert cands.meta["intent"] == "clarification_request"
    assert question.split("?")[0][-30:] in last.content or "?" in last.content


def test_clarification_the_keyword_rules_miss_is_taken_from_the_analyzer(client):
    def ears(si, msgs):
        out = HANDLERS["turn_analyzer"](si)
        if "last job" in si.get("answer", ""):
            out["intent"] = "clarification_request"
        return out
    routing.set_provider_override("simulated", SimulatedProvider(overrides={"turn_analyzer": ears}))
    try:
        c, sid = started(client)
        c.turn(sid, STRONG_ANSWER)
        r = c.turn(sid, "Talking about my last job here I suppose, the one at the bank").json()
        from interview_intelligence.db.models import InterviewMessage
        from interview_intelligence.db.session import db_session
        from sqlalchemy import select
        with db_session() as db:
            last = db.execute(select(InterviewMessage).where(InterviewMessage.role == "interviewer")
                              .order_by(InterviewMessage.seq.desc())).scalars().first()
        assert last.action == "CLARIFY_QUESTION", last.action
        assert r["messages"][-1]["content"]
    finally:
        routing.set_provider_override("simulated", None)


def test_repeat_after_a_follow_up_repeats_the_follow_up(client):
    c, sid = started(client, difficulty="hard")
    c.turn(sid, STRONG_ANSWER)
    probe = c.turn(sid, WEAK_ANSWER).json()["messages"][-1]["content"]
    r = c.turn(sid, "Sorry, can you repeat that?").json()
    assert probe.rstrip("?")[-25:] in r["messages"][-1]["content"], (probe, r["messages"][-1]["content"])


def test_a_planned_question_that_repeats_an_asked_one_is_skipped():
    b = policy_bp()
    for sec in b["sections"]:
        for it in sec["items"]:
            if it["qid"] == "Q3":
                it["text"] = "Tell me about a time you had to influence a stakeholder without authority."
            if it["qid"] == "Q4":
                it["text"] = "Describe a situation where you influenced stakeholders without having authority over them."
            if it["qid"] == "Q5":
                it["text"] = "How would you value a mid-sized FMCG business with falling margins?"
    s = policy_st(b, qid="Q3")
    s["asked"] = ["Q1", "Q2", "Q3"]
    act = P.choose_next(s, b)
    assert act.qid == "Q5" and any("repeats Q3" in r for r in act.reasons), (act.qid, act.reasons)


def test_a_short_answer_misheard_as_an_unrelated_question_is_still_an_answer(client):
    def ears(si, msgs):
        out = HANDLERS["turn_analyzer"](si)
        out["intent"] = "off_topic_question"
        return out
    routing.set_provider_override("simulated", SimulatedProvider(overrides={"turn_analyzer": ears}))
    try:
        c, sid = started(client)
        c.turn(sid, STRONG_ANSWER)
        c.turn(sid, "I negotiated the contract myself and cut the renewal cost by 12% in one quarter.")
        from interview_intelligence.db.models import InterviewMessage
        from interview_intelligence.db.session import db_session
        from sqlalchemy import select
        with db_session() as db:
            last = db.execute(select(InterviewMessage).where(InterviewMessage.role == "interviewer")
                              .order_by(InterviewMessage.seq.desc())).scalars().first()
        assert last.action != "REDIRECT", "a statement was treated as an unrelated question and the question re-asked"
    finally:
        routing.set_provider_override("simulated", None)


def test_a_clarifying_question_is_never_quoted_as_evidence(client):
    from tests.conftest import run_jobs
    c, sid = started(client)
    c.turn(sid, STRONG_ANSWER)
    c.turn(sid, "Do you mean in my current role?")
    c.turn(sid, STRONG_ANSWER)
    client.post(f"/v1/sessions/{sid}/end", headers=c.h)
    run_jobs()
    rep = client.get(f"/v1/sessions/{sid}/report", headers=c.h).json()["report"]
    flat = str(rep["questions"]) + str([e for comp in rep["competencies"] for e in comp["evidence"]])
    assert "Do you mean in my current role" not in flat
