"""Two active sessions per user (spec §5), including the race between simultaneous starts."""

import threading

import pytest

from tests.conftest import IS_PG, enable_pro, run_jobs
from tests.helpers import Candidate


def test_third_session_blocked_and_slot_frees(client):
    enable_pro(client)
    c = Candidate(client)
    cv = c.upload_cv()
    jd = c.paste_jd()
    run_jobs()
    s1 = c.create(cv["id"], jd["id"]).json()["id"]
    s2 = c.create(cv["id"], jd["id"], mode="hr_behavioral").json()["id"]
    r3 = c.create(cv["id"], jd["id"], mode="functional")
    assert r3.status_code == 409
    err = r3.json()["error"]
    assert err["code"] == "active_limit"
    assert "already have 2 active interview sessions" in err["message"]
    assert {a["id"] for a in err["active_sessions"]} == {s1, s2}
    # abandoning frees a slot
    assert client.post(f"/v1/sessions/{s1}/abandon", headers=c.h).status_code == 200
    r4 = c.create(cv["id"], jd["id"], mode="functional")
    assert r4.status_code == 201
    # completing frees a slot too
    run_jobs()
    sid = r4.json()["id"]
    client.post(f"/v1/sessions/{sid}/start", headers=c.h)
    client.post(f"/v1/sessions/{sid}/end", headers=c.h)
    assert c.create(cv["id"], jd["id"], mode="case").status_code == 201
    me = client.get("/v1/me", headers=c.h).json()
    assert me["limits"]["active_count"] == 2


def test_limit_is_configurable_by_admin(client):
    enable_pro(client)
    from tests.conftest import ADMIN_EMAIL, auth, mint
    client.patch("/v1/admin/config", json={"values": {"limits.max_active_sessions": 1}},
                 headers=auth(mint(email=ADMIN_EMAIL, tier="free")))
    c = Candidate(client)
    cv, jd = c.upload_cv(), c.paste_jd()
    assert c.create(cv["id"], jd["id"]).status_code == 201
    assert c.create(cv["id"], jd["id"]).status_code == 409


@pytest.mark.skipif(not IS_PG, reason="row-lock race only meaningful on Postgres")
def test_parallel_creates_never_exceed_limit(client):
    enable_pro(client)
    c = Candidate(client)
    cv, jd = c.upload_cv(), c.paste_jd()
    run_jobs()
    results = []
    barrier = threading.Barrier(6)

    def go():
        barrier.wait()
        results.append(c.create(cv["id"], jd["id"]).status_code)

    threads = [threading.Thread(target=go) for _ in range(6)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(results).count(201) == 2, results
    assert all(s in (201, 409) for s in results), results


def test_daily_limit(client):
    enable_pro(client)
    from tests.conftest import ADMIN_EMAIL, auth, mint
    admin_h = auth(mint(email=ADMIN_EMAIL, tier="free"))
    client.patch("/v1/admin/config", json={"values": {"limits.max_sessions_per_day": 1,
                                                      "limits.max_active_sessions": 5}}, headers=admin_h)
    c = Candidate(client)
    cv, jd = c.upload_cv(), c.paste_jd()
    assert c.create(cv["id"], jd["id"]).status_code == 201
    r = c.create(cv["id"], jd["id"])
    assert r.status_code == 429 and r.json()["error"]["code"] == "daily_limit"
