"""Session state machine (spec §5) and timeout sweeps."""

import itertools
from datetime import timedelta

import pytest

from interview_intelligence.interview_engine import lifecycle as L
from tests.conftest import enable_pro, run_jobs
from tests.helpers import STRONG_ANSWER, Candidate


def test_transition_matrix_is_exhaustive_and_terminal_states_are_final():
    states = list(L.TRANSITIONS)
    assert set(states) == {"created", "uploading", "analyzing", "ready", "active", "paused", "completed",
                           "abandoned", "expired", "failed"}
    for s in L.TERMINAL:
        assert L.TRANSITIONS[s] == set()
    for a, b in itertools.product(states, states):
        assert L.can_transition(a, b) == (b in L.TRANSITIONS[a])
    assert not L.can_transition("completed", "active")
    assert not L.can_transition("ready", "completed")
    assert L.can_transition("paused", "active")


def _age(sid, **delta):
    from interview_intelligence.db.models import InterviewSession, utcnow
    from interview_intelligence.db.session import db_session
    with db_session() as db:
        s = db.get(InterviewSession, sid)
        s.last_activity_at = utcnow() - timedelta(**delta)


def test_sweep_timeouts(client):
    import uuid
    enable_pro(client)
    c = Candidate(client)
    sid = c.ready_session()
    _age(uuid.UUID(sid), hours=25)
    me = client.get("/v1/me", headers=c.h).json()
    # lazy sweep runs on the next create; trigger it via maintenance
    from interview_intelligence.jobs.maintenance import sweep_all
    sweep_all()
    s = client.get(f"/v1/sessions/{sid}", headers=c.h).json()
    assert s["status"] == "expired"


def test_idle_active_pauses_then_expires_and_is_assessed(client):
    import uuid
    enable_pro(client)
    c = Candidate(client)
    sid = c.ready_session()
    client.post(f"/v1/sessions/{sid}/start", headers=c.h)
    c.turn(sid, STRONG_ANSWER)
    c.turn(sid, STRONG_ANSWER)
    c.turn(sid, STRONG_ANSWER)
    from interview_intelligence.jobs.maintenance import sweep_all
    _age(uuid.UUID(sid), minutes=30)
    sweep_all()
    assert client.get(f"/v1/sessions/{sid}", headers=c.h).json()["status"] == "paused"
    _age(uuid.UUID(sid), hours=30)
    sweep_all()
    s = client.get(f"/v1/sessions/{sid}", headers=c.h).json()
    assert s["status"] == "expired" and s["assessment_status"] == "pending"
    run_jobs()
    rep = client.get(f"/v1/sessions/{sid}/report", headers=c.h).json()
    assert rep["report"]["headline"]["assessment_confidence"] == "limited"


def test_cannot_start_twice_or_answer_before_start(client):
    enable_pro(client)
    c = Candidate(client)
    sid = c.ready_session()
    r = c.turn(sid, "hello")
    assert r.status_code == 409
    assert client.post(f"/v1/sessions/{sid}/start", headers=c.h).status_code == 200
    again = client.post(f"/v1/sessions/{sid}/start", headers=c.h)
    assert again.status_code == 200 and len(again.json()["messages"]) >= 1  # resume view, not a new interview
    client.post(f"/v1/sessions/{sid}/end", headers=c.h)
    assert c.turn(sid, "late answer").status_code == 409
    assert client.post(f"/v1/sessions/{sid}/start", headers=c.h).status_code == 409


def test_end_before_start_abandons_slot(client):
    enable_pro(client)
    c = Candidate(client)
    sid = c.ready_session()
    r = client.post(f"/v1/sessions/{sid}/end", headers=c.h)
    assert r.json()["session"]["status"] == "abandoned"
