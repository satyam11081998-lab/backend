"""The full product path with the offline provider: upload -> analysis -> blueprint -> live
interview -> evidence -> evaluation -> feedback -> report -> progress."""

from tests.conftest import enable_pro, run_jobs
from tests.helpers import STRONG_ANSWER, WEAK_ANSWER, Candidate


def test_full_interview_flow(client):
    enable_pro(client)
    c = Candidate(client)
    sid = c.ready_session()

    s = client.get(f"/v1/sessions/{sid}", headers=c.h).json()
    pre = s["pre_interview_summary"]
    # the candidate sees the role as understood, never the plan (sections, counts, claims to test)
    assert pre["role"]["title"] and not {"sections", "claims_to_investigate", "not_planned", "competencies_identified"} & set(pre)
    import uuid
    from interview_intelligence.db.models import InterviewSession
    from interview_intelligence.db.session import db_session
    with db_session() as db:
        full = db.get(InterviewSession, uuid.UUID(sid)).pre_interview_summary
    assert full["competencies_identified"] >= 5
    assert full["strong_in_cv"] + full["need_validation"] == full["competencies_identified"]
    assert any(sec["kind"] == "intro" for sec in full["sections"])

    r = client.post(f"/v1/sessions/{sid}/start", headers=c.h)
    assert r.status_code == 200, r.text
    first = r.json()["messages"][0]
    assert first["role"] == "interviewer" and "?" in first["content"]

    answers = [STRONG_ANSWER, WEAK_ANSWER, STRONG_ANSWER, "Because I owned the forecast, I chose to delay the launch by "
               "two weeks; it cost us 3% of the quarter's volume but avoided a 15% stock-out risk.", STRONG_ANSWER] * 4
    last = None
    for a in answers:
        r = c.turn(sid, a)
        assert r.status_code == 200, r.text
        body = r.json()
        last = body
        # never any evaluation signal in the live payload
        flat = str(body).lower()
        assert "score" not in flat and "band" not in flat and "evidence_state" not in flat
        if body["session"]["status"] == "completed":
            break
    if last["session"]["status"] != "completed":
        r = client.post(f"/v1/sessions/{sid}/end", headers=c.h)
        assert r.status_code == 200
    run_jobs()
    r = client.get(f"/v1/sessions/{sid}/report", headers=c.h)
    assert r.status_code == 200, r.text
    rep = r.json()["report"]
    for key in ("header", "headline", "executive_assessment", "health", "competencies", "role_alignment",
                "questions", "coverage_map", "preparation_plan", "interviewer_learned", "transparency"):
        assert key in rep, key
    assert rep["headline"]["assessment_confidence"] in ("high", "moderate", "limited")
    assert all(c_["band"] for c_ in rep["competencies"])
    # every scored competency cites evidence that exists
    refs = {e["ref"] for q in rep["questions"] for e in [{"ref": x} for x in q["evidence_refs"]]}
    for comp in rep["competencies"]:
        if comp["score"] is not None:
            assert comp["evidence"], comp["competency_id"]
            assert {e["ref"] for e in comp["evidence"]} <= refs
        else:
            assert comp["evidence_state"] == "not_sufficiently_tested"
    # "why was I asked this"
    q = rep["questions"][0]
    w = client.get(f"/v1/sessions/{sid}/questions/{q['exchange_id']}/why", headers=c.h)
    assert w.status_code == 200 and "selected because" in w.json()["why_asked"]
    # transcript available after the interview
    t = client.get(f"/v1/sessions/{sid}/transcript", headers=c.h)
    assert t.status_code == 200 and len(t.json()["messages"]) > 4
    # progress endpoint works
    pr = client.get("/v1/progress", headers=c.h)
    assert pr.status_code == 200 and pr.json()["trajectories"]


def test_documents_are_reused_across_interviews(client):
    enable_pro(client)
    c = Candidate(client)
    cv1 = c.upload_cv()
    cv2 = c.upload_cv()
    assert cv1["id"] == cv2["id"], "same file must dedupe"
    jd = c.paste_jd()
    run_jobs()
    r1 = c.create(cv1["id"], jd["id"], mode="hr_behavioral")
    r2 = c.create(cv1["id"], jd["id"], mode="functional")
    assert r1.status_code == 201 and r2.status_code == 201
    from interview_intelligence.db.models import DocumentAnalysis
    from interview_intelligence.db.session import db_session
    run_jobs()
    with db_session() as db:
        n = db.query(DocumentAnalysis).count()
    assert n == 2, "CV and JD analysed once each, reused by both sessions"
