"""Round 7 (tester + owner feedback, 2026-10-03):
 * the candidate never sees the plan (question kinds, counts, CV claims to test) before or during it;
 * the interviewer opens like a person: who it is, the mode, the length and the running order;
 * more breadth: the whole CV (not one job), the person beyond it (hobbies, positions of
   responsibility) and business awareness built on real recent news — never invented news;
 * plans: one free 15-minute interview per account, Ultra with a fair-use cap, a plans page only
   people with Interview Intelligence can see; and a hard stop at the planned length."""

import uuid

import pytest
from sqlalchemy import select

from interview_intelligence import host
from interview_intelligence.ai import routing
from interview_intelligence.ai.runner import RunContext
from interview_intelligence.ai.simulated import HANDLERS, SimulatedProvider
from interview_intelligence.db.models import InterviewBlueprint, InterviewSession, InterviewState
from interview_intelligence.db.session import db_session
from interview_intelligence.interview_engine import blueprint as B
from interview_intelligence.interview_engine.modes import InterviewConfig, allocate_sections
from interview_intelligence.question_engine import selector
from interview_intelligence.question_engine.generator import generate_for_section
from interview_intelligence.question_engine.schemas import PlannedQuestion
from tests.conftest import ADMIN_EMAIL, CV_TEXT, auth, enable_pro, mint, run_jobs
from tests.helpers import STRONG_ANSWER, Candidate

ADMIN = None

CV_WITH_LIFE = CV_TEXT + """Positions of responsibility
- General Secretary, Marketing Club, IMI Delhi: ran a 2-day case competition with 40 teams
Extra-curricular
- Finalist, national case competition 2021
Interests: long-distance running, chess, Carnatic music
"""

NEWS = [
    {"title": "Snack makers raise prices as palm oil costs climb", "summary": "Several FMCG companies said they "
     "would take price increases on packaged snacks this quarter.", "source_name": "Business Daily",
     "published_at": "2026-09-28T05:00:00+00:00", "category": "business", "keywords": ["fmcg", "snacks", "pricing"],
     "gd_worthiness_score": 8},
    {"title": "Central bank holds rates steady", "summary": "The policy rate was left unchanged.",
     "source_name": "Markets Today", "published_at": "2026-09-30T05:00:00+00:00", "category": "economy",
     "keywords": ["rates"], "gd_worthiness_score": 9},
]


def _admin_h():
    return auth(mint(email=ADMIN_EMAIL, tier="free"))


def _bp(sid):
    with db_session() as db:
        return db.execute(select(InterviewBlueprint).where(InterviewBlueprint.session_id == uuid.UUID(sid))
                          ).scalar_one().blueprint


def _grant(client, email, grant_type="test"):
    r = client.post("/v1/admin/access-grants", json={"email": email, "grant_type": grant_type}, headers=_admin_h())
    assert r.status_code == 201, r.text
    return r.json()


# --------------------------------------------------------------------------- the plan stays hidden
def test_candidate_never_sees_the_plan(client):
    enable_pro(client)
    c = Candidate(client)
    sid = c.ready_session()
    s = client.get(f"/v1/sessions/{sid}", headers=c.h).json()
    assert set(s["pre_interview_summary"]) <= {"role", "jd_quality", "jd_warnings", "cv_warnings", "company_context"}
    r = client.post(f"/v1/sessions/{sid}/start", headers=c.h).json()
    for payload in (r["session"], client.get(f"/v1/sessions/{sid}/room", headers=c.h).json()["progress"],
                    c.turn(sid, STRONG_ANSWER).json()["session"]):
        assert not {"section_title", "questions_planned", "questions_asked"} & set(payload), payload
        assert payload["section"] in ("interview", "closing")


# --------------------------------------------------------------------------- the opening
def test_opening_says_who_mode_length_and_running_order(client):
    enable_pro(client)
    c = Candidate(client)
    sid = c.ready_session(mode="grill", difficulty="hard", duration_minutes=30)
    first = client.post(f"/v1/sessions/{sid}/start", headers=c.h).json()["messages"][0]["content"]
    assert first.startswith("Hi, I'm your MECE interviewer.")
    assert "grill mode" in first and "30 minutes" in first and "Brand Manager" in first
    assert "push for specifics" in first
    assert first.rstrip().endswith("?"), "it still asks the first question"
    from interview_intelligence.interview_engine.interviewer import AGENDA_PHRASES
    bp = _bp(sid)
    kinds = [a["kind"] for a in bp["agenda"]]
    assert kinds == [s["kind"] for s in bp["sections"] if s["kind"] not in ("intro", "closing") and s["items"]]
    spoken = [AGENDA_PHRASES[k].split("{m}")[0] for k in kinds]
    positions = [first.find(p) for p in spoken]
    assert all(p >= 0 for p in positions) and positions == sorted(positions), (first, spoken)
    for k, phrase in AGENDA_PHRASES.items():
        if k not in kinds:
            assert phrase.split("{m}")[0] not in first, f"announced a part that is not in the plan: {k}"


def test_repeating_the_first_question_does_not_replay_the_whole_introduction(client):
    enable_pro(client)
    c = Candidate(client)
    sid = c.ready_session()
    client.post(f"/v1/sessions/{sid}/start", headers=c.h)
    again = c.turn(sid, "Sorry, can you repeat that?").json()["messages"][-1]["content"]
    assert "MECE interviewer" not in again and "walk me through your background" in again


# --------------------------------------------------------------------------- breadth
def test_a_mixed_interview_covers_the_person_and_business_awareness(client):
    enable_pro(client)
    c = Candidate(client)
    cv = c.upload_cv(text=CV_WITH_LIFE)
    jd = c.paste_jd()
    run_jobs()
    sid = c.create(cv["id"], jd["id"], mode="mixed", duration_minutes=45).json()["id"]
    run_jobs()
    bp = _bp(sid)
    kinds = [s["kind"] for s in bp["sections"]]
    assert "personal" in kinds and "awareness" in kinds, kinds
    personal = next(s for s in bp["sections"] if s["kind"] == "personal")["items"][0]["text"]
    assert any(w in personal.lower() for w in ("running", "chess", "carnatic", "marketing club", "case competition")), \
        personal
    awareness = next(s for s in bp["sections"] if s["kind"] == "awareness")["items"][0]
    # no news source in this process: the candidate picks the story; the interviewer states no news
    assert not awareness["selection_reason"].get("news_ids") and bp["news"] == []
    assert kinds.index("cv") < kinds.index("personal") < kinds.index("behavioral"), "a natural running order"


def test_business_awareness_uses_real_recent_news_when_the_host_has_it(client):
    enable_pro(client)
    host.set_news_provider(lambda: NEWS)
    try:
        c = Candidate(client)
        sid = c.ready_session(mode="mixed", duration_minutes=45)
    finally:
        host.set_news_provider(None)
    bp = _bp(sid)
    assert bp["news"] and bp["news"][0]["title"] == NEWS[0]["title"], "the snacks story fits a snacks brand role best"
    item = next(s for s in bp["sections"] if s["kind"] == "awareness")["items"][0]
    assert item["selection_reason"]["news_ids"] == ["N1"] and "palm oil" in item["text"]


def test_a_broken_news_source_never_breaks_preparation(client):
    enable_pro(client)

    def boom():
        raise RuntimeError("news table missing")
    host.set_news_provider(boom)
    try:
        sid = Candidate(client).ready_session(mode="mixed", duration_minutes=45)
    finally:
        host.set_news_provider(None)
    assert _bp(sid)["news"] == []


def test_political_and_geopolitical_news_is_never_used():
    rows = [{"title": "Ruling party wins state election", "category": "policy"},
            {"title": "Ceasefire talks stall at the border", "category": "geopolitics"},
            {"title": "Minister says party will review fuel prices", "category": "business"},
            {"title": "Quick-commerce firms cut delivery fees to win snack orders", "category": "business"}]
    picked = B.pick_news(rows, {"identity": {"title": "Brand Manager - Snacks"}}, "Brand", "fmcg")
    assert [n["title"] for n in picked] == ["Quick-commerce firms cut delivery fees to win snack orders"]


def test_a_question_that_adds_facts_to_the_news_is_rejected():
    def inventive(si, msgs):
        return {"questions": [{"text": "Snack makers raised prices by 40% this quarter. What would you do?",
                               "archetype_id": "news_take", "competency_ids": ["commercial_judgment"],
                               "question_type": "awareness", "intent": "x", "expected_evidence": ["a", "b"],
                               "probe_tree": ["a?", "b?"], "news_ids": ["N1"]}]}
    routing.set_provider_override("simulated", SimulatedProvider(overrides={"question_generator": inventive}))
    try:
        out = generate_for_section(
            section_kind="awareness", n=1, competencies=[{"competency_id": "commercial_judgment", "name": "x"}],
            rubrics={}, role={"identity": {"title": "Brand Manager"}}, seniority="mid", difficulty_level=3,
            technical_depth=2, claims=[], company_facts=[], focus_areas=[], avoid=[], ctx=RunContext(),
            news=B.pick_news(NEWS, {"identity": {"title": "Brand Manager"}}, "Brand", "fmcg"))
    finally:
        routing.set_provider_override("simulated", None)
    assert out == [], "a number that is not in the headline is an invented fact"


def test_cv_questions_spread_across_roles_not_one_job():
    claims = [{"id": "C1", "experience_id": "X1"}, {"id": "C2", "experience_id": "X1"}, {"id": "C3", "experience_id": "X1"},
              {"id": "C4", "experience_id": "X2"}, {"id": "C5", "experience_id": None, "type": "award"}]
    assert [c["id"] for c in B._diverse_claims(claims, 3)] == ["C1", "C4", "C5"]
    assert [c["id"] for c in B._diverse_claims(claims, 5)] == ["C1", "C4", "C5", "C2", "C3"]


def test_short_interviews_keep_few_areas_long_ones_add_breadth():
    def kinds(minutes, mode="mixed"):
        cfg = InterviewConfig(mode=mode, duration_minutes=minutes)
        return [p.kind for p in allocate_sections(cfg, {}, False, True, False, 270)]
    short, full = kinds(15), kinds(45)
    assert len(short) - 2 <= 3, short
    assert {"personal", "awareness"} <= set(full), full
    assert full.index("cv") < full.index("personal") < full.index("functional")


def test_the_same_kind_of_question_is_not_picked_twice():
    def q(text, arch):
        return PlannedQuestion(text=text, archetype_id=arch, competency_ids=["ownership"], section_kind="behavioral",
                               origin="curated")
    pool = [q("Tell me about a time you owned a problem end to end.", "ownership_deep_dive"),
            q("Think of a decision you got wrong and what you changed afterwards.", "failure_reflection")]
    chosen, _ = selector.select(pool, 1, need={"ownership": 1.0}, difficulty_level=3, avoid=[],
                                used_archetypes=["ownership_deep_dive"])
    assert chosen[0].archetype_id == "failure_reflection"


# --------------------------------------------------------------------------- hard stop
def test_the_interview_stops_at_its_length(client):
    enable_pro(client)
    c = Candidate(client)
    sid = c.ready_session(duration_minutes=15)
    client.post(f"/v1/sessions/{sid}/start", headers=c.h)
    c.turn(sid, STRONG_ANSWER)
    with db_session() as db:
        st = db.execute(select(InterviewState).where(InterviewState.session_id == uuid.UUID(sid))).scalar_one()
        state = dict(st.state)
        state["clock"] = {**state["clock"], "active_s": 15 * 60 + 200}
        st.state = state
    r = c.turn(sid, STRONG_ANSWER).json()
    assert r["session"]["status"] == "completed"
    assert "out of time" in r["messages"][-1]["content"]
    with db_session() as db:
        assert db.get(InterviewSession, uuid.UUID(sid)).ended_reason == "time_limit"


# --------------------------------------------------------------------------- plans
def test_free_interview_is_15_minutes_once_and_the_report_stays(client):
    _grant(client, "trial@example.invalid", "trial")
    c = Candidate(client, email="trial@example.invalid", tier="free")
    me = client.get("/v1/me", headers=c.h).json()
    assert me["access"]["via"] == "trial" and me["limits"]["allowed_durations"] == [15]
    assert me["plan"]["trial"]["available"] is True and me["plan"]["visible"] is True

    cv, jd = c.upload_cv(), c.paste_jd()
    run_jobs()
    a = c.create(cv["id"], jd["id"], duration_minutes=45).json()
    b = c.create(cv["id"], jd["id"]).json()
    run_jobs()
    assert a["duration_minutes"] == 15 and a["plan"] == "trial", "the free interview has one length"
    assert client.post(f"/v1/sessions/{a['id']}/start", headers=c.h).status_code == 200
    # one interview, ever: the other prepared one cannot start, and no new one can be prepared
    r = client.post(f"/v1/sessions/{b['id']}/start", headers=c.h)
    assert r.status_code == 409 and r.json()["error"]["code"] == "trial_used"
    client.post(f"/v1/sessions/{b['id']}/abandon", headers=c.h)
    r = c.create(cv["id"], jd["id"])
    assert r.status_code == 409 and r.json()["error"]["code"] == "trial_used"
    # ...but the one in progress can be finished
    assert c.turn(a["id"], STRONG_ANSWER).status_code == 200
    assert client.post(f"/v1/sessions/{a['id']}/end", headers=c.h).status_code == 200
    run_jobs()
    me = client.get("/v1/me", headers=c.h).json()
    assert me["access"]["allowed"] is False and me["has_history"] is True
    assert me["plan"]["trial"]["used"] is True and me["plan"]["visible"] is True
    assert client.get(f"/v1/sessions/{a['id']}/report", headers=c.h).status_code == 200
    assert client.get("/v1/plans", headers=c.h).status_code == 200, "a used free interview leads to the plans"
    nav = client.get("/v1/access", headers=c.h).json()
    assert nav["allowed"] is False and nav["nav"] is True, "the menu entry stays: the report lives there"
    r = c.turn(a["id"], STRONG_ANSWER)
    assert r.status_code == 403


def test_a_trial_account_cannot_prepare_endless_interviews(client):
    _grant(client, "trial2@example.invalid", "trial")
    c = Candidate(client, email="trial2@example.invalid", tier="free")
    cv, jd = c.upload_cv(), c.paste_jd()
    run_jobs()
    for _ in range(3):
        sid = c.create(cv["id"], jd["id"]).json()["id"]
        client.post(f"/v1/sessions/{sid}/abandon", headers=c.h)
    r = c.create(cv["id"], jd["id"])
    assert r.status_code == 429 and r.json()["error"]["code"] == "trial_prepare_limit"


def test_the_free_interview_opens_to_everyone_only_when_switched_on(client):
    c = Candidate(client, email="someone@example.invalid", tier="free")
    assert client.get("/v1/me", headers=c.h).json()["access"]["allowed"] is False
    assert client.get("/v1/plans", headers=c.h).status_code == 404, "the plans page is not public"
    client.patch("/v1/admin/config", json={"values": {"plans.trial_open": True}}, headers=_admin_h())
    me = client.get("/v1/me", headers=c.h).json()
    assert me["access"]["allowed"] is True and me["access"]["via"] == "trial"


def test_ultra_has_a_fair_use_cap(client):
    _grant(client, "ultra@example.invalid", "ultra")
    client.patch("/v1/admin/config", json={"values": {"plans.ultra_monthly_interviews": 1}}, headers=_admin_h())
    c = Candidate(client, email="ultra@example.invalid", tier="free")
    me = client.get("/v1/me", headers=c.h).json()
    assert me["access"]["via"] == "ultra" and me["plan"]["ultra"]["left"] == 1
    sid = c.ready_session(duration_minutes=60)
    assert client.post(f"/v1/sessions/{sid}/start", headers=c.h).status_code == 200
    client.post(f"/v1/sessions/{sid}/end", headers=c.h)
    r = c.create(*[d["id"] for d in (c.upload_cv(), c.paste_jd())])
    assert r.status_code == 429 and r.json()["error"]["code"] == "ultra_monthly_limit"


def test_a_future_mece_ultra_tier_is_recognised(client):
    c = Candidate(client, email="paid@example.invalid", tier="ultra")
    assert client.get("/v1/me", headers=c.h).json()["access"]["via"] == "ultra"


def test_plans_page_and_interest_are_for_people_with_access_only(client):
    _grant(client, "tester@example.invalid")
    c = Candidate(client, email="tester@example.invalid", tier="free")
    p = client.get("/v1/plans", headers=c.h).json()
    assert p["preview"] is True and p["ultra"]["price_inr"] == 1299 and p["trial"]["minutes"] == 15
    for _ in range(2):  # recorded once per account
        assert client.post("/v1/plans/interest", json={"plan": "ultra"}, headers=c.h).json()["interested"] is True
    assert client.get("/v1/plans", headers=c.h).json()["ultra"]["interested"] is True
    a = client.get("/v1/admin/plans", headers=_admin_h()).json()
    assert a["interest"]["ultra"] == 1 and a["interest"]["recent"][0]["email"] == "tester@example.invalid"
    stranger = Candidate(client, email="stranger@example.invalid", tier="pro")
    assert client.post("/v1/plans/interest", json={"plan": "ultra"}, headers=stranger.h).status_code == 404
    client.patch("/v1/admin/config", json={"values": {"plans.visibility": "off"}}, headers=_admin_h())
    assert client.get("/v1/plans", headers=c.h).status_code == 404
    assert client.get("/v1/plans", headers=_admin_h()).status_code == 200, "admins can always preview it"


def test_admin_grant_types_and_plan_settings_are_validated(client):
    g = _grant(client, "x@example.invalid", "trial")
    assert g["grant_type"] == "trial"
    r = client.patch(f"/v1/admin/access-grants/{g['id']}", json={"grant_type": "ultra"}, headers=_admin_h())
    assert r.json()["grant_type"] == "ultra"
    assert client.patch(f"/v1/admin/access-grants/{g['id']}", json={"grant_type": "gold"},
                        headers=_admin_h()).status_code == 422
    assert client.patch("/v1/admin/config", json={"values": {"plans.trial_minutes": 900}},
                        headers=_admin_h()).status_code == 422
    assert client.patch("/v1/admin/config", json={"values": {"plans.visibility": "public"}},
                        headers=_admin_h()).status_code == 422


def test_free_interviews_can_run_on_a_cheaper_voice_engine(client):
    client.patch("/v1/admin/config", json={"values": {"voice.engine": "realtime",
                                                      "plans.trial_voice_engine": "standard"}}, headers=_admin_h())
    _grant(client, "cheap@example.invalid", "trial")
    c = Candidate(client, email="cheap@example.invalid", tier="free")
    assert client.get("/v1/me", headers=c.h).json()["flags"]["voice_engine"] == "standard"
    sid = c.ready_session()
    client.post(f"/v1/sessions/{sid}/start", headers=c.h)
    r = client.post("/v1/voice/live", json={"session_id": sid}, headers=c.h)
    assert r.status_code == 403 and r.json()["error"]["code"] == "live_off"
    # everyone else keeps the global engine
    _grant(client, "full@example.invalid")
    full = Candidate(client, email="full@example.invalid", tier="free")
    assert client.get("/v1/me", headers=full.h).json()["flags"]["voice_engine"] == "realtime"
