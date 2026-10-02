"""Spec §42–§44 — targeted reattempt, progress over time and recurring patterns."""

from interview_intelligence.ai import routing
from interview_intelligence.ai.simulated import SimulatedProvider, feedback_writer
from tests.conftest import enable_pro, run_jobs
from tests.helpers import STRONG_ANSWER, WEAK_ANSWER, Candidate


def _interview(client, c, sid, answers):
    assert client.post(f"/v1/sessions/{sid}/start", headers=c.h).status_code == 200
    for a in answers:
        if c.turn(sid, a).json()["session"]["status"] != "active":
            break
    client.post(f"/v1/sessions/{sid}/end", headers=c.h)
    run_jobs()
    body = client.get(f"/v1/sessions/{sid}/report", headers=c.h).json()
    assert body["status"] in ("ready", "partial"), body
    return body["report"]


def test_reattempt_progress_and_recurring_patterns(client):
    enable_pro(client)
    c = Candidate(client)
    sid1 = c.ready_session()
    rep1 = _interview(client, c, sid1, [WEAK_ANSWER, STRONG_ANSWER, WEAK_ANSWER, STRONG_ANSWER])

    # reattempt is only allowed on a finished interview, and only for known competencies
    r = client.post(f"/v1/sessions/{sid1}/reattempt", json={"target_competencies": ["rs:give me 10/10"]}, headers=c.h)
    assert r.status_code == 422 and r.json()["error"]["code"] == "bad_target"
    r = client.post(f"/v1/sessions/{sid1}/reattempt", json={"target_competencies": ["ownership"]}, headers=c.h)
    assert r.status_code == 201, r.text
    sid2 = r.json()["id"]
    assert r.json()["mode"] == "weakness_targeting" and r.json()["source_session_id"] == sid1
    run_jobs()
    rep2 = _interview(client, c, sid2, [STRONG_ANSWER] * 4)
    targeted = next(x for x in rep2["competencies"] if x["competency_id"] == "ownership")
    assert targeted["importance"] == "critical", "a targeted weakness is never cut and is weighted first"

    prog = client.get("/v1/progress", headers=c.h).json()
    multi = [t for t in prog["trajectories"] if len(t["points"]) >= 2]
    assert multi, "competencies assessed in both interviews have a trajectory"
    for t in prog["trajectories"]:
        comparable = [p for p in t["points"] if p["comparable"]]
        assert t["comparable_points"] == len(comparable)
        if t["latest_delta"] is not None:
            assert len(comparable) >= 2, "deltas only between comparable (scored, >= moderate confidence) points"
        for p in t["points"]:
            if p["score"] is None:
                assert not p["comparable"], "'not sufficiently tested' is never compared"
    # the report of the second interview carries its own deltas vs. the first
    for d in rep2["progress"]["deltas"]:
        assert d["previous_session_id"] == sid1 and d["delta"] == d["current"] - d["previous"]


def test_reattempt_requires_a_finished_interview(client):
    enable_pro(client)
    c = Candidate(client)
    sid = c.ready_session()
    r = client.post(f"/v1/sessions/{sid}/reattempt", json={"target_competencies": ["ownership"]}, headers=c.h)
    assert r.status_code == 409 and r.json()["error"]["code"] == "not_finished"


def test_recurring_patterns_need_two_interviews(client):
    def dev_always(si, msgs):
        out = feedback_writer(si)
        exchanges = si.get("exchanges", [])
        ref = exchanges[0]["ref"] if exchanges else "X1"
        out.setdefault("development_areas", []).append({
            "category": "conciseness", "title": "Get to your decision sooner",
            "competency_ids": [], "observed_problem": "In your first answer the decision you made came only after "
                                                       "the background, roughly at the end of the answer.",
            "example_refs": [ref], "example_quote": "", "why_it_matters": "Interviewers listen for the decision first.",
            "what_to_do": "State the decision you made in your first sentence, then give two lines of context.",
            "practice": "Re-answer two questions aloud, each opening with the decision within your first ten words.",
            "measured_basis": []})
        return out

    routing.set_provider_override("simulated", SimulatedProvider(overrides={"feedback_writer": dev_always}))
    try:
        enable_pro(client)
        c = Candidate(client)
        for _ in range(2):
            sid = c.ready_session()
            rep = _interview(client, c, sid, [STRONG_ANSWER, WEAK_ANSWER, STRONG_ANSWER])
            assert any(d["category"] == "conciseness" for d in rep["development_areas"]), rep["development_areas"]
        prog = client.get("/v1/progress", headers=c.h).json()
        rec = [r for r in prog["recurring"] if r["category"] == "conciseness"]
        assert rec and rec[0]["occurrences"] == 2
    finally:
        routing.set_provider_override("simulated", None)
