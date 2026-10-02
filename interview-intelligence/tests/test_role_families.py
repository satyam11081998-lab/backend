"""Spec §116 — representative runs across eight role families, each in a different mode,
difficulty and duration. Uses the offline simulated models, so this checks STRUCTURE
(classification, competency model, blueprint shape and question QA, report adaptivity,
evidence grounding, fairness), not judgement quality — that is qa/run_golden.py's job."""

import itertools
import json
import re
import uuid

import pytest
from sqlalchemy import select

from interview_intelligence.db.models import InterviewBlueprint
from interview_intelligence.db.session import db_session
from interview_intelligence.interview_engine.modes import MODES
from interview_intelligence.question_engine import quality
from interview_intelligence.role_taxonomy.library import library
from interview_intelligence.textutil import jaccard
from tests.conftest import enable_pro, run_jobs
from tests.helpers import Candidate
from tests.role_fixtures import PLANS, ROLES

PROTECTED = re.compile(r"\b(accent|gender|religio\w*|caste|married|marital|pregnan\w*|nationality|ethnic\w*|"
                       r"disabilit\w*|age \d+|years old)\b", re.IGNORECASE)


def _blueprint(sid):
    with db_session() as db:
        return db.execute(select(InterviewBlueprint).where(InterviewBlueprint.session_id == uuid.UUID(sid))
                          ).scalar_one().blueprint


def _norm(s):
    return re.sub(r"\s+", " ", s or "").strip().lower()


@pytest.mark.parametrize("family", sorted(ROLES))
def test_role_family_run(client, family):
    enable_pro(client)
    role, (mode, difficulty, minutes) = ROLES[family], PLANS[family]
    lib = library()
    fam = lib.family(family)
    c = Candidate(client)
    cv = c.upload_cv(text=role["cv"])
    jd = c.paste_jd(text=role["jd"])
    run_jobs()
    r = c.create(cv["id"], jd["id"], mode=mode, difficulty=difficulty, duration_minutes=minutes)
    assert r.status_code == 201, r.text
    sid = r.json()["id"]
    run_jobs()
    s = client.get(f"/v1/sessions/{sid}", headers=c.h).json()
    assert s["status"] == "ready", s
    assert s["role_family"] == family

    # ---------------- blueprint
    bp = _blueprint(sid)
    comp_ids = {x["competency_id"] for x in bp["competencies"]}
    assert set(fam.defaults) <= comp_ids, "family defaults are always in the competency model"
    issues = bp["qa"]["deterministic"]
    gaps = [i for i in issues if i["issue"] == "coverage"]
    assert [i for i in issues if i["issue"] != "coverage"] == [], issues
    if minutes >= 30:
        assert gaps == [], ("every critical/high competency gets a planned question", gaps, bp["qa"]["coverage_repair"])
    else:  # a 15-minute interview cannot hold one question per key competency: disclosed, not hidden
        assert all(g["unresolved_reason"] == "no_redundant_slot" for g in gaps)
        assert len(s["pre_interview_summary"]["not_planned"]) == len(gaps)
    assert sum(sec["budget_s"] for sec in bp["sections"]) <= minutes * 60
    kinds = [sec["kind"] for sec in bp["sections"]]
    assert kinds[0] == "intro" and kinds[-1] == "closing"
    items = [it for sec in bp["sections"] for it in sec["items"]]
    for it in items:
        assert quality.problems(it["text"]) == [], (it["text"], quality.problems(it["text"]))
        if it["origin"] != "fixed":
            assert it["intent"] and it["competency_ids"] and it["expected_evidence"], it
            assert set(it["competency_ids"]) <= comp_ids
    for a, b in itertools.combinations([it["text"] for it in items], 2):
        assert jaccard(a, b) < 0.6, (a, b)
    if mode == "hr_behavioral":
        assert not {"functional", "technical", "case"} & set(kinds)
    if mode == "case":
        assert "case" in kinds
    if mode == "technical":
        assert ("technical" in kinds) == fam.technical
    if not fam.technical:
        assert "technical" not in kinds, "non-technical roles get deep functional questions instead"
    if mode in ("cv_jd", "cv_attack", "cv_deep_dive", "grill"):
        assert "cv" in kinds and any(it["origin"] == "cv_specific" for it in items)
    assert bp["difficulty_vector"]["max_probes"] == (4 if difficulty == "grill" else
                                                     max(1, bp["difficulty_vector"]["max_probes"]))

    # ---------------- interview
    assert client.post(f"/v1/sessions/{sid}/start", headers=c.h).status_code == 200
    for i in range(5):
        r = c.turn(sid, role["answer"] if i % 2 == 0 else "We worked on it as a team and it went fine.")
        assert r.status_code == 200
        if r.json()["session"]["status"] != "active":
            break
    client.post(f"/v1/sessions/{sid}/end", headers=c.h)
    run_jobs()
    rep_body = client.get(f"/v1/sessions/{sid}/report", headers=c.h).json()
    assert rep_body["status"] in ("ready", "partial"), rep_body
    rep = rep_body["report"]

    # ---------------- report adapts to the role
    assert rep["header"]["role_family"] == fam.name
    cats = [h["category"] for h in rep["health"]]
    assert set(cats) - {"Role-specific knowledge", "Other competencies"} <= set(fam.report_dimensions)
    assert set(cats) & set(fam.report_dimensions), "health uses this family's dimensions"
    assert {a["competency_id"] for a in rep["competencies"]} == comp_ids

    # ---------------- every quote is verbatim from what the candidate said
    said = _norm(" ".join(m["content"] for m in client.get(f"/v1/sessions/{sid}/transcript", headers=c.h)
                          .json()["messages"] if m["role"] == "candidate"))
    for comp in rep["competencies"]:
        assert (comp["score"] is None) == (comp["evidence_state"] == "not_sufficiently_tested")
        for e in comp["evidence"]:
            assert _norm(e["quote"]).rstrip(".") in said, e["quote"]
    assert not PROTECTED.search(json.dumps(rep)), PROTECTED.search(json.dumps(rep)).group(0)


def test_report_dimensions_differ_between_families():
    lib = library()
    dims = {f: tuple(lib.family(f).report_dimensions) for f in ROLES}
    assert len(set(dims.values())) >= 6, dims
    for f in ROLES:
        assert lib.family(f).id == f
    assert all(m in MODES for m, _, _ in PLANS.values())


def test_question_library_grows_without_personal_data(client):
    """Generated questions are kept for reuse/analytics; CV-derived ones never leave the session."""
    from interview_intelligence.db.models import GeneratedQuestion
    from tests.helpers import Candidate as _C
    enable_pro(client)
    for _ in range(2):
        c = _C(client)
        c.ready_session()
    with db_session() as db:
        rows = db.execute(select(GeneratedQuestion)).scalars().all()
    assert rows, "generated questions are recorded"
    assert all("You mention" not in r.text and r.origin == "generated" for r in rows)
    assert any(r.times_selected >= 2 for r in rows), "the same generated question is counted, not duplicated"
