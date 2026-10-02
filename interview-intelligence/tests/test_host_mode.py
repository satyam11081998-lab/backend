"""Host mode: II mounted inside another FastAPI app (the MECE backend) under /ii.

The host supplies only an identity resolver (Authorization header -> HostIdentity). These tests
use a stand-in host with its own route and CORS layer, and a fake resolver whose tokens look
like  "Bearer host:<uuid>:<tier>:<admin 0|1>:<guest 0|1>:<email>".
"""

from __future__ import annotations

import uuid

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from tests.conftest import ADMIN_EMAIL, DB_URL, mint, run_jobs
from tests.helpers import STRONG_ANSWER, Candidate

CALLS = {"n": 0}


def fake_resolver(authorization):
    from interview_intelligence.host import HostIdentity
    CALLS["n"] += 1
    tok = (authorization or "")[7:].strip()
    parts = tok.split(":")
    if len(parts) != 6 or parts[0] != "host":
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    _, uid, tier, adm, guest, email = parts
    if tier == "banned":
        raise HTTPException(status_code=403, detail="Account suspended")
    return HostIdentity(user_id=uid, email=email, tier=tier, is_admin=adm == "1", is_guest=guest == "1")


def htok(uid=None, *, tier="pro", adm=False, guest=False, email="pro@example.invalid") -> dict:
    return {"Authorization": f"Bearer host:{uid or uuid.uuid4()}:{tier}:{int(adm)}:{int(guest)}:{email}"}


class HostCandidate(Candidate):
    @property
    def h(self):
        return htok(self.uid, tier=self.tier, adm=self.adm, email=self.email)


def make_host(events=None):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def host_lifespan(app):
        if events is not None:
            events.append("host-start")
        yield
        if events is not None:
            events.append("host-stop")

    host = FastAPI(lifespan=host_lifespan)
    host.add_middleware(CORSMiddleware, allow_origins=["https://mece.in"], allow_credentials=True,
                        allow_methods=["*"], allow_headers=["*"])

    @host.get("/ping")
    def ping():
        return {"pong": True}

    return host


@pytest.fixture
def host_env(monkeypatch):
    from interview_intelligence.auth.assertion import reset_host_cache
    from interview_intelligence.host import set_identity_resolver
    monkeypatch.setenv("II_DATABASE_URL", DB_URL)
    CALLS["n"] = 0
    reset_host_cache()
    yield
    set_identity_resolver(None)  # never leak host mode into the rest of the suite
    reset_host_cache()


@pytest.fixture
def hclient(host_env):
    from interview_intelligence.host import mount
    host = make_host()
    mount(host, "/ii", fake_resolver)
    with TestClient(host, base_url="http://testserver/ii") as c:
        yield c


def host_enable_pro(c):
    r = c.patch("/v1/admin/config", json={"values": {"ii.enabled_for_pro": True}},
                headers=htok(tier="free", adm=True, email="owner@example.invalid"))
    assert r.status_code == 200, r.text


def test_dormant_without_database_url(host_env, monkeypatch):
    from interview_intelligence.host import mount
    monkeypatch.delenv("II_DATABASE_URL", raising=False)
    host = make_host()
    lazy = mount(host, "/ii", fake_resolver)
    with TestClient(host) as c:
        r = c.get("/ii/v1/me", headers=htok())
        assert r.status_code == 503 and r.json()["error"]["code"] == "not_configured"
        assert c.get("/ping").json() == {"pong": True}  # the host is unaffected
    assert lazy._app is None  # nothing was imported or built


def test_host_routes_and_lifespan_unaffected(host_env):
    from interview_intelligence.host import mount
    events: list = []
    host = make_host(events)
    lazy = mount(host, "/ii", fake_resolver)
    stopped = []
    lazy.stop = lambda: stopped.append(True)  # type: ignore[method-assign]
    with TestClient(host) as c:
        assert events == ["host-start"]
        assert c.get("/ping").json() == {"pong": True}
        assert lazy._app is None  # lazy: not loaded until someone uses II
        assert c.get("/ii/healthz").json()["ok"] is True
        assert lazy._app is not None
    assert events == ["host-start", "host-stop"] and stopped == [True]


def test_identity_errors(hclient):
    r = hclient.get("/v1/me")
    assert r.status_code == 401 and r.json()["error"]["code"] == "missing_token"
    r = hclient.get("/v1/me", headers={"Authorization": "Bearer nonsense"})
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_token"
    r = hclient.get("/v1/me", headers=htok(tier="banned"))
    assert r.status_code == 403
    r = hclient.get("/v1/me", headers=htok(guest=True))
    assert r.status_code == 403 and r.json()["error"]["code"] == "guest"
    r = hclient.get("/v1/me", headers={"Authorization": "Bearer host:not-a-uuid:pro:0:0:x@example.invalid"})
    assert r.status_code == 401


def test_c10_assertions_are_not_accepted_in_host_mode(hclient):
    # In host mode identity comes only from the host's own session check.
    r = hclient.get("/v1/me", headers={"Authorization": f"Bearer {mint()}"})
    assert r.status_code == 401


def test_access_rules_hold_in_host_mode(hclient):
    free = HostCandidate(hclient, tier="free", email="free@example.invalid")
    r = hclient.post("/v1/documents/text", json={"kind": "jd", "text": "x" * 400}, headers=free.h)
    assert r.status_code == 403 and r.json()["error"]["code"] == "not_entitled"
    pro = HostCandidate(hclient, tier="pro")
    r = hclient.post("/v1/documents/text", json={"kind": "jd", "text": "x" * 400}, headers=pro.h)
    assert r.status_code == 403  # private preview: launch flag still off
    # MECE admins are II admins; an admin can grant test access to a free account by email
    owner = htok(tier="free", adm=True, email="owner@example.invalid")
    r = hclient.post("/v1/admin/access-grants", json={"email": "Free@Example.invalid"}, headers=owner)
    assert r.status_code == 201, r.text
    me = hclient.get("/v1/me", headers=free.h).json()
    assert me["access"]["allowed"] is True
    # a non-admin cannot reach admin routes
    assert hclient.get("/v1/admin/overview", headers=pro.h).status_code == 403
    # the configured II admin email still works too
    assert hclient.get("/v1/admin/overview", headers=htok(tier="free", email=ADMIN_EMAIL)).status_code == 200


def test_identity_is_cached_briefly(hclient):
    h = htok(tier="pro")
    for _ in range(5):
        assert hclient.get("/v1/me", headers=h).status_code == 200
    assert CALLS["n"] == 1


def test_full_flow_through_the_host(hclient):
    host_enable_pro(hclient)
    c = HostCandidate(hclient)
    sid = c.ready_session()
    r = hclient.post(f"/v1/sessions/{sid}/start", headers=c.h)
    assert r.status_code == 200, r.text
    for _ in range(4):
        r = c.turn(sid, STRONG_ANSWER)
        assert r.status_code == 200, r.text
        if r.json()["session"]["status"] == "completed":
            break
    if r.json()["session"]["status"] != "completed":
        assert hclient.post(f"/v1/sessions/{sid}/end", headers=c.h).status_code == 200
    run_jobs()
    r = hclient.get(f"/v1/sessions/{sid}/report", headers=c.h)
    assert r.status_code == 200, r.text
    assert r.json()["report"]["competencies"]
    # another user cannot see it
    other = HostCandidate(hclient, email="other@example.invalid")
    assert hclient.get(f"/v1/sessions/{sid}/report", headers=other.h).status_code == 404


def test_single_cors_layer(hclient):
    r = hclient.options("/v1/me", headers={"Origin": "https://mece.in", "Access-Control-Request-Method": "GET",
                                           "Access-Control-Request-Headers": "authorization"})
    assert r.status_code == 200
    assert r.headers.get_list("access-control-allow-origin") == ["https://mece.in"]
    r = hclient.get("/v1/me", headers={**htok(), "Origin": "https://mece.in"})
    assert r.headers.get_list("access-control-allow-origin") == ["https://mece.in"]


def test_start_failure_never_breaks_the_host(host_env, monkeypatch):
    from interview_intelligence import main as ii_main
    from interview_intelligence.host import mount

    def boom():
        raise RuntimeError("no encryption key")

    monkeypatch.setattr(ii_main, "start_background", boom)
    host = make_host()
    lazy = mount(host, "/ii", fake_resolver)
    with TestClient(host) as c:
        r = c.get("/ii/v1/me", headers=htok())
        assert r.status_code == 503 and r.json()["error"]["code"] == "unavailable"
        assert c.get("/ping").json() == {"pong": True}
        # backs off instead of retrying on every request
        monkeypatch.setattr(ii_main, "start_background", lambda: [])
        assert c.get("/ii/v1/me", headers=htok()).status_code == 503
        lazy._failed_at -= lazy.RETRY_AFTER_S + 1
        assert c.get("/ii/v1/me", headers=htok()).status_code == 200


def test_ai_and_drive_keys_fall_back_to_the_hosts_names(monkeypatch):
    from interview_intelligence.config import load_settings
    for k in ("II_OPENAI_API_KEY", "II_GROQ_API_KEY", "II_GEMINI_API_KEY", "II_ANTHROPIC_API_KEY",
              "II_GDRIVE_CLIENT_ID", "II_GDRIVE_ROOT_FOLDER_ID", "GOOGLE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "host-openai")
    monkeypatch.setenv("GROQ_API_KEY", "host-groq")
    monkeypatch.setenv("GEMINI_API_KEY", "host-gemini")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "host-anthropic")
    monkeypatch.setenv("GOOGLE_DRIVE_CLIENT_ID", "host-drive")
    s = load_settings()
    assert (s.openai_api_key, s.groq_api_key, s.gemini_api_key, s.anthropic_api_key, s.gdrive_client_id) == \
        ("host-openai", "host-groq", "host-gemini", "host-anthropic", "host-drive")
    assert s.drive_configured is False  # Drive stays off without II's own root folder
    monkeypatch.setenv("II_OPENAI_API_KEY", "ii-openai")
    assert load_settings().openai_api_key == "ii-openai"  # an II-only key always wins
    monkeypatch.setenv("II_DB_POOL_SIZE", "3")
    assert load_settings().db_pool_size == 3


def test_idle_worker_wakes_when_a_job_is_enqueued():
    import threading
    import time

    from interview_intelligence.db.session import db_session
    from interview_intelligence.jobs import queue

    done = threading.Event()
    queue.register("test.wake")(lambda db, payload: done.set())
    w = queue.Worker(threads=1, poll_s=0.05, idle_poll_s=30.0, active_window_s=0.5)
    w.start()
    try:
        time.sleep(0.8)  # past the active window: the worker is now in a 30 s idle wait
        with db_session() as db:
            queue.enqueue(db, "test.wake", {})
        assert done.wait(5), "an idle worker should wake on an in-process enqueue"
    finally:
        w.stop()
    for t in w._ts:
        t.join(3)
        assert not t.is_alive(), "stop() must break the idle wait"
