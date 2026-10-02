"""Live progress while the person waits: real stages reported by the work itself, real findings,
and an honest estimate only for how far the CURRENT model call is along."""

import time

from interview_intelligence.jobs import progress
from tests.conftest import enable_pro, run_jobs
from tests.helpers import STRONG_ANSWER, Candidate


def test_tracker_steps_percent_and_facts(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr(progress, "_now", lambda: clock[0])
    progress.start("k", [("a", "Step A", 10.0), ("b", "Step B", 30.0)])
    progress.begin("k", "a")
    v = progress.view("k")
    assert [s["state"] for s in v["steps"]] == ["active", "todo"] and v["pct"] == 0
    clock[0] += 5
    assert progress.view("k")["pct"] == round(100 * 5 / 40)
    clock[0] += 500  # a step that overruns never shows as finished
    assert progress.view("k")["pct"] == round(100 * 10 * progress.STEP_CAP / 40)
    progress.fact("k", "3 roles, 2 organisations")
    progress.begin("k", "b")
    progress.detail("k", "Section 1 of 2")
    v = progress.view("k")
    assert v["steps"][0]["state"] == "done" and v["steps"][0]["facts"] == ["3 roles, 2 organisations"]
    assert v["steps"][1]["detail"] == "Section 1 of 2" and v["step_eta_s"] == 30.0 and v["pct"] == 25
    progress.portion("k", 2, 3)  # 2 of 3 parts really done beats the time estimate
    assert progress.view("k")["pct"] == round(100 * (10 + 30 * 2 / 3) / 40)
    progress.insert("k", [("c", "Step C", 10.0)])
    assert [s["id"] for s in progress.view("k")["steps"]] == ["a", "b", "c"]
    assert progress.view("k")["pct"] < 100
    progress.finish("k")
    v = progress.view("k")
    assert v["done"] and v["pct"] == 100 and all(s["state"] == "done" for s in v["steps"])
    assert progress.view("nothing") is None
    # nothing here may ever break the work it describes
    progress.begin("nothing", "x")
    progress.fact("nothing", "y")
    progress.finish("nothing")


def test_eta_comes_from_recent_real_runs(client):
    from interview_intelligence.db.models import ModelRun
    from interview_intelligence.db.session import db_session as session_scope
    assert progress.eta_for(None, "cv_parser") == progress.DEFAULT_ETA_S["cv_parser"]
    with session_scope() as db:
        for ms in (9000, 11000, 10000, 50):
            db.add(ModelRun(stage="cv_parse", provider="openai", model="m", prompt_id="cv_parser", latency_ms=ms,
                            status="ok"))
        db.add(ModelRun(stage="cv_parse", provider="simulated", model="s", prompt_id="cv_parser", latency_ms=1,
                        status="ok"))
    progress.reset_for_tests()
    with session_scope() as db:
        assert progress.eta_for(db, "cv_parser") == 10.0  # median of the real runs; the simulator is ignored
        assert progress.eta_for(db, "cv_parser", calls=3) == 30.0


def test_document_progress_shows_real_steps_then_findings(client):
    enable_pro(client)
    c = Candidate(client)
    up = c.upload_cv()
    p = up["progress"]
    assert [s["id"] for s in p["steps"]] == ["read", "private", "analyse"]
    assert p["steps"][0]["state"] == "done" and "words" in p["steps"][0]["facts"][0]
    assert p["steps"][1]["state"] == "done" and p["steps"][1]["facts"]  # contact details hidden (CV has an email)
    assert any("contact detail" in f for f in p["steps"][1]["facts"])
    assert p["steps"][2]["state"] == "waiting" and p["steps"][2]["looking_for"] and 0 < p["pct"] < 100

    # while the worker is analysing it, the step is live with an estimate the page can animate
    progress.start(f"doc:{up['id']}", [("analyse", "x", 20.0)])
    progress.begin(f"doc:{up['id']}", "analyse")
    p = client.get(f"/v1/documents/{up['id']}", headers=c.h).json()["progress"]
    assert p["steps"][2]["state"] == "active" and p["step_eta_s"] == 20.0 and p["step_weight"] > 0.5

    run_jobs()
    d = client.get(f"/v1/documents/{up['id']}", headers=c.h).json()
    p = d["progress"]
    assert d["analysis_status"] == "ready" and p["done"] and p["pct"] == 100
    found = " ".join(p["steps"][2]["facts"])
    assert "role" in found and "achievement" in found, found
    assert p["highlights"], "a few skills to show as chips"

    jd = c.paste_jd()
    run_jobs()
    p = client.get(f"/v1/documents/{jd['id']}", headers=c.h).json()["progress"]
    assert [s["id"] for s in p["steps"]] == ["read", "analyse"]  # nothing personal in a JD
    assert any("must-have" in f for f in p["steps"][1]["facts"]), p["steps"][1]["facts"]
    # the list view stays light
    docs = client.get("/v1/documents", headers=c.h).json()["documents"]
    assert all("progress" not in x for x in docs)


def test_interview_build_reports_each_stage_with_findings(client):
    enable_pro(client)
    c = Candidate(client)
    cv, jd = c.upload_cv(), c.paste_jd()
    run_jobs()
    r = c.create(cv["id"], jd["id"])
    sid = r.json()["id"]
    s = client.get(f"/v1/sessions/{sid}", headers=c.h).json()
    assert "prep_progress" in s and s["prep_progress"] is None  # queued: the page shows "starting"
    run_jobs()
    v = progress.view(f"prep:{sid}")
    assert v["done"] and [x["id"] for x in v["steps"]] == ["inputs", "role", "match", "rubrics", "questions", "check"]
    facts = {x["id"]: " ".join(x["facts"]) for x in v["steps"]}
    assert "this role needs" in facts["match"] and ("backed by your CV" in facts["match"] or "each one" in facts["match"])
    assert "question" in facts["questions"] and "section" in facts["questions"]
    assert facts["check"]
    s = client.get(f"/v1/sessions/{sid}", headers=c.h).json()
    assert s["status"] == "ready" and "prep_progress" not in s


def test_report_progress_is_reported_and_carries_no_scores(client):
    enable_pro(client)
    c = Candidate(client)
    sid = c.ready_session(duration_minutes=15)
    assert client.post(f"/v1/sessions/{sid}/start", headers=c.h).status_code == 200
    for _ in range(3):
        c.turn(sid, STRONG_ANSWER)
    client.post(f"/v1/sessions/{sid}/end", headers=c.h)
    progress.start(f"report:{sid}", [("evidence", "x", 5.0)])
    progress.begin(f"report:{sid}", "evidence")
    r = client.get(f"/v1/sessions/{sid}/report", headers=c.h)
    assert r.status_code == 202 and r.json()["progress"]["steps"][0]["state"] == "active"
    run_jobs()
    v = progress.view(f"report:{sid}")
    assert v["done"] and [x["id"] for x in v["steps"]] == ["evidence", "score", "check", "feedback", "assemble"]
    flat = str(v).lower()
    assert "evidence from" in flat and "assessed" in flat
    assert "score:" not in flat and "/10" not in flat and "band" not in flat
    assert client.get(f"/v1/sessions/{sid}/report", headers=c.h).status_code == 200
